#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info(f"Downloading input artifact: {args.input_artifact} at {artifact_local_path}")

    # Load the dataset
    df = pd.read_csv(artifact_local_path)
    logger.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    # drop price outliers
    df = df[(df["price"] >= args.min_price) & (df["price"] <= args.max_price)]
    logger.info(f"Drop price outliers, e.g. rows with price < min_price or > max_price. After cleaning, dataset has {df.shape[0]} rows and {df.shape[1]} columns")
    # convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    logger.info("Converted last_review to datetime")
    # save the cleaned dataset
    df.to_csv("clean_sample.csv", index=False)

    # Create an output artifact and log it to W&B
    logger.info(f"Creating output artifact: {args.output_artifact} of type {args.output_type} with description '{args.output_description}'")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact to download in W&B",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact to create in W&B",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="The description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to keep in the dataset.",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to keep in the dataset.",
        required=True
    )


    args = parser.parse_args()

    go(args)

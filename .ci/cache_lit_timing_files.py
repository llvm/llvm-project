# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Caches .lit_test_times.txt files between premerge invocations.

.lit_test_times.txt files are used by lit to order tests to best take advantage
of parallelism. Having them around and up to date can result in a ~15%
improvement in test times. This script downloading cached test time files and
uploading new versions to the GCS buckets used for caching.
"""

import sys
import os
import logging
import multiprocessing.pool
import pathlib
import platform
import glob

from google.cloud import storage

GCS_PARALLELISM = 100


def _maybe_upload_timing_file(bucket, timing_file_path):
    blob_prefix = f"lit_timing_{platform.system().lower()}/"
    if os.path.exists(timing_file_path):
        timing_file_blob = bucket.blob(blob_prefix + timing_file_path)
        timing_file_blob.upload_from_filename(timing_file_path)


def upload_timing_files(storage_client, bucket_name: str):
    bucket = storage_client.bucket(bucket_name)
    with multiprocessing.pool.ThreadPool(GCS_PARALLELISM) as thread_pool:
        futures = []
        for timing_file_path in glob.glob("**/.lit_test_times.txt", recursive=True):
            futures.append(
                thread_pool.apply_async(
                    _maybe_upload_timing_file, (bucket, timing_file_path)
                )
            )
        for future in futures:
            future.get()
    print("Done uploading")


def _maybe_download_timing_file(blob):
    file_name = blob.name.removeprefix("lit_timing/")
    pathlib.Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(file_name)


def download_timing_files(storage_client, bucket_name: str):
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix="lit_timing")
    with multiprocessing.pool.ThreadPool(GCS_PARALLELISM) as thread_pool:
        futures = []
        for timing_file_blob in blobs:
            futures.append(
                thread_pool.apply_async(
                    _maybe_download_timing_file, (timing_file_blob,)
                )
            )
        for future in futures:
            future.get()
    print("Done downloading")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.fatal("Expected usage is cache_lit_timing_files.py <upload/download>")
        sys.exit(1)
    action = sys.argv[1]
    storage_client = storage.Client()
    bucket_name = os.environ["CACHE_GCS_BUCKET"]
    if action == "download":
        download_timing_files(storage_client, bucket_name)
    elif action == "upload":
        upload_timing_files(storage_client, bucket_name)
    else:
        logging.fatal("Expected usage is cache_lit_timing_files.py <upload/download>")
        sys.exit(1)

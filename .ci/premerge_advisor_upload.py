# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Script for uploading results to the premerge advisor."""

import argparse
import os
import platform
import sys

import requests

import generate_test_report_lib

PREMERGE_ADVISOR_URL = (
    "http://premerge-advisor.premerge-advisor.svc.cluster.local:5000/upload"
)


def main(commit_sha, workflow_run_number, build_log_files):
    junit_objects, ninja_logs = generate_test_report_lib.load_info_from_files(
        build_log_files
    )
    test_failures = generate_test_report_lib.get_failures(junit_objects)
    source = "pull_request" if "GITHUB_ACTIONS" in os.environ else "postcommit"
    failure_info = {
        "source_type": source,
        "base_commit_sha": commit_sha,
        "source_id": workflow_run_number,
        "failures": [],
    }
    if test_failures:
        for name, failure_message in test_failures:
            failure_info["failures"].append({"name": name, "message": failure_message})
    else:
        ninja_failures = generate_test_report_lib.find_failure_in_ninja_logs(ninja_logs)
        for name, failure_message in ninja_failures:
            failure_info["failures"].append({"name": name, "message": failure_message})
    requests.post(PREMERGE_ADVISOR_URL, json=failure_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("commit_sha", help="The base commit SHA for the test.")
    parser.add_argument("workflow_run_number", help="The run number from GHA.")
    parser.add_argument(
        "build_log_files", help="Paths to JUnit report files and ninja logs.", nargs="*"
    )
    args = parser.parse_args()

    # Skip uploading results on AArch64 for now because the premerge advisor
    # service is not available on AWS currently.
    if platform.machine() == "arm64":
        sys.exit(0)

    main(args.commit_sha, args.workflow_run_number, args.build_log_files)

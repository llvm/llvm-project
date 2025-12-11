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

# These are IP addresses of the two premerge advisor instances. They should
# eventually be updated to domain names.
PREMERGE_ADVISOR_URLS = [
    "http://34.82.126.63:5000/upload",
    "http://136.114.125.23:5000/upload",
]


def main(commit_sha, workflow_run_number, build_log_files):
    junit_objects, ninja_logs = generate_test_report_lib.load_info_from_files(
        build_log_files
    )
    test_failures = generate_test_report_lib.get_failures(junit_objects)
    source = "pull_request" if "GITHUB_ACTIONS" in os.environ else "postcommit"
    current_platform = f"{platform.system()}-{platform.machine()}".lower()
    failure_info = {
        "source_type": source,
        "base_commit_sha": commit_sha,
        "source_id": workflow_run_number,
        "failures": [],
        "platform": current_platform,
    }
    if test_failures:
        for _, failures in test_failures.items():
            for name, failure_message in failures:
                failure_info["failures"].append(
                    {"name": name, "message": failure_message}
                )
    else:
        ninja_failures = generate_test_report_lib.find_failure_in_ninja_logs(ninja_logs)
        for name, failure_message in ninja_failures:
            failure_info["failures"].append({"name": name, "message": failure_message})
    for premerge_advisor_url in PREMERGE_ADVISOR_URLS:
        requests.post(premerge_advisor_url, json=failure_info, timeout=5)


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
    if platform.machine() == "arm64" or platform.machine() == "aarch64":
        sys.exit(0)

    main(args.commit_sha, args.workflow_run_number, args.build_log_files)

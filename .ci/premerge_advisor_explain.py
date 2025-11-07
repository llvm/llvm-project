# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Script for getting explanations from the premerge advisor."""

import argparse
import os
import platform
import sys

import requests

import generate_test_report_lib

PREMERGE_ADVISOR_URL = (
    "http://premerge-advisor.premerge-advisor.svc.cluster.local:5000/explain"
)


def main(commit_sha: str, build_log_files: list[str]):
    junit_objects, ninja_logs = generate_test_report_lib.load_info_from_files(
        build_log_files
    )
    test_failures = generate_test_report_lib.get_failures(junit_objects)
    current_platform = f"{platform.system()}-{platform.machine()}".lower()
    explanation_request = {
        "base_commit_sha": commit_sha,
        "platform": current_platform,
        "failures": [],
    }
    if test_failures:
        for _, failures in test_failures.items():
            for name, failure_messsage in failures:
                explanation_request["failures"].append(
                    {"name": name, "message": failure_messsage}
                )
    else:
        ninja_failures = generate_test_report_lib.find_failure_in_ninja_logs(ninja_logs)
        for name, failure_message in ninja_failures:
            explanation_request["failures"].append(
                {"name": name, "message": failure_message}
            )
    advisor_response = requests.get(
        PREMERGE_ADVISOR_URL, json=explanation_request, timeout=5
    )
    if advisor_response.status_code == 200:
        print(advisor_response.json())
    else:
        print(advisor_response.reason)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("commit_sha", help="The base commit SHA for the test.")
    parser.add_argument(
        "build_log_files", help="Paths to JUnit report files and ninja logs.", nargs="*"
    )
    args = parser.parse_args()

    # Skip looking for results on AArch64 for now because the premerge advisor
    # service is not available on AWS currently.
    if platform.machine() == "arm64":
        sys.exit(0)

    main(args.commit_sha, args.build_log_files)

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Script to generate a build report for buildkite."""

import argparse
import os
import subprocess

import generate_test_report_lib


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "title", help="Title of the test report, without Markdown formatting."
    )
    parser.add_argument("context", help="Annotation context to write to.")
    parser.add_argument("return_code", help="The build's return code.", type=int)
    parser.add_argument("junit_files", help="Paths to JUnit report files.", nargs="*")
    args = parser.parse_args()

    # All of these are required to build a link to download the log file.
    env_var_names = [
        "BUILDKITE_ORGANIZATION_SLUG",
        "BUILDKITE_PIPELINE_SLUG",
        "BUILDKITE_BUILD_NUMBER",
        "BUILDKITE_JOB_ID",
    ]
    buildkite_info = {k: v for k, v in os.environ.items() if k in env_var_names}
    if len(buildkite_info) != len(env_var_names):
        buildkite_info = None

    report, style = generate_test_report_lib.generate_report_from_files(
        args.title, args.return_code, args.junit_files, buildkite_info
    )

    if report:
        p = subprocess.Popen(
            [
                "buildkite-agent",
                "annotate",
                "--context",
                args.context,
                "--style",
                style,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        # The report can be larger than the buffer for command arguments so we send
        # it over stdin instead.
        _, err = p.communicate(input=report)
        if p.returncode:
            raise RuntimeError(f"Failed to send report to buildkite-agent:\n{err}")

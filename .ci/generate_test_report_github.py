# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Script to generate a build report for Github."""

import argparse

import generate_test_report_lib


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("return_code", help="The build's return code.", type=int)
    parser.add_argument(
        "build_test_logs", help="Paths to JUnit report files and ninja logs.", nargs="*"
    )
    args = parser.parse_args()

    report = generate_test_report_lib.generate_report_from_files(
        generate_test_report_lib.compute_platform_title(),
        args.return_code,
        args.build_test_logs,
    )

    print(report)

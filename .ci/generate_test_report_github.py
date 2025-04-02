# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Script to generate a build report for Github."""

import argparse

import generate_test_report_lib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "title", help="Title of the test report, without Markdown formatting."
    )
    parser.add_argument("return_code", help="The build's return code.", type=int)
    parser.add_argument("junit_files", help="Paths to JUnit report files.", nargs="*")
    args = parser.parse_args()

    report, _ = generate_test_report_lib.generate_report_from_files(
        args.title, args.return_code, args.junit_files, None
    )

    print(report)

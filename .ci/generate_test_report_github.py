# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Script to generate a build report for Github."""

import argparse
import platform

import generate_test_report_lib

PLATFORM_TITLES = {
    "Windows": ":window: Windows x64 Test Results",
    "Linux": ":penguin: Linux x64 Test Results",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("return_code", help="The build's return code.", type=int)
    parser.add_argument("junit_files", help="Paths to JUnit report files.", nargs="*")
    args = parser.parse_args()

    report = generate_test_report_lib.generate_report_from_files(
        PLATFORM_TITLES[platform.system()], args.return_code, args.junit_files
    )

    print(report)

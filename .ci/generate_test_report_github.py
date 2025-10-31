# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Script to generate a build report for Github."""

import argparse
import platform

import generate_test_report_lib

def compute_platform_title() -> str:
    logo = ":window:" if platform.system() == "Windows" else ":penguin:"
    # On Linux the machine value is x86_64 on Windows it is AMD64.
    if platform.machine() == "x86_64" or platform.machine() == "AMD64":
        arch = "x64"
    else:
        arch = platform.machine()
    return f"{logo} {platform.system()} {arch} Test Results"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("return_code", help="The build's return code.", type=int)
    parser.add_argument(
        "build_test_logs", help="Paths to JUnit report files and ninja logs.", nargs="*"
    )
    args = parser.parse_args()

    report = generate_test_report_lib.generate_report_from_files(
        compute_platform_title(), args.return_code, args.build_test_logs
    )

    print(report)

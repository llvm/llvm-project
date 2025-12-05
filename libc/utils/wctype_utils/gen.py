#!/usr/bin/env python3
#
# ===- Run wctype generator ----------------------------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==------------------------------------------------------------------------==#

from conversion.gen_conversion_data import extract_maps_from_unicode_file
from conversion.hex_writer import write_hex_conversions
from data.fetch import fetch_unicode_data_files
from sys import argv
from sys import exit


def write_wctype_conversion_data(llvm_project_root_path: str) -> None:
    """Generates and writes wctype conversion data files"""
    lower_to_upper, upper_to_lower = extract_maps_from_unicode_file(
        f"{llvm_project_root_path}/libc/utils/wctype_utils/data/UnicodeData.txt"
    )
    write_hex_conversions(
        file_path=f"{llvm_project_root_path}/libc/src/__support/wctype/lower_to_upper.inc",
        mappings=lower_to_upper,
    )
    write_hex_conversions(
        file_path=f"{llvm_project_root_path}/libc/src/__support/wctype/upper_to_lower.inc",
        mappings=upper_to_lower,
    )


def main() -> None:
    if len(argv) < 2:
        print("Codegen: wctype data generator script")
        print(f"Usage:\n\t{argv[0]} <path-to-llvm-project-root> [--fetch-only]")
        print("Options:")
        print(
            "\t--fetch-only\tFetches necessary unicode data files only with no generation."
        )
        exit(1)

    if len(argv) == 3 and argv[2] == "--fetch-only":
        fetch_unicode_data_files(llvm_project_root_path=argv[1])
        print("Fetched necessary unicode data files.")
    else:
        write_wctype_conversion_data(llvm_project_root_path=argv[1])
        print(
            f"wctype conversion data is written to {argv[1]}/libc/src/__support/wctype/"
        )


if __name__ == "__main__":
    main()

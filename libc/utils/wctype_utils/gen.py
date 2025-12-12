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
from sys import argv
from sys import exit


def write_wctype_conversion_data(
    llvm_project_root_path: str, unicode_data_folder_path: str
) -> None:
    """Generates and writes wctype conversion data files"""
    lower_to_upper, upper_to_lower = extract_maps_from_unicode_file(
        f"{unicode_data_folder_path}/UnicodeData.txt"
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
    if len(argv) != 3:
        print("Codegen: wctype data generator script")
        print(
            f"Usage:\n\t{argv[0]} <path-to-llvm-project-root> <path-to-unicode-data-folder>"
        )
        print(
            "INFO: You can download Unicode data files from https://www.unicode.org/Public/UCD/latest/ucd/"
        )
        exit(1)

    write_wctype_conversion_data(
        llvm_project_root_path=argv[1], unicode_data_folder_path=argv[2]
    )
    print(f"wctype conversion data is written to {argv[1]}/libc/src/__support/wctype/")


if __name__ == "__main__":
    main()

# ===- Generate conversion data for wctype utils -------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==------------------------------------------------------------------------==#


def extract_maps_from_unicode_file(
    file_path: str,
) -> tuple[dict[int, int], dict[int, int]]:
    """Extracts lower-to-upper and upper-to-lower case mappings"""
    lower_to_upper = {}
    upper_to_lower = {}

    # Construct upper-lower case mappings
    with open(file_path) as file:
        for line in file.readlines():
            line_entries = line.split(";")
            code_point, name, classification = line_entries[:3]
            code_point = int(code_point, 16)

            if classification == "Lu":
                if line_entries[13]:
                    upper_to_lower[code_point] = int(line_entries[13], 16)
            elif classification == "Ll":
                if line_entries[12]:
                    lower_to_upper[code_point] = int(line_entries[12], 16)

    return (lower_to_upper, upper_to_lower)

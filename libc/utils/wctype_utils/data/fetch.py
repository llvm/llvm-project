#!/usr/bin/env python3
#
# ===- Fetch files necessary for wctype generator ------------*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==------------------------------------------------------------------------==#
#
# This file is meant to be run manually by maintainers to fetch the latest
# unicode data files from unicode.org necessary for generating wctype data.
# All rights to the data belong to unicode.org.

from urllib.request import urlretrieve


def fetch_unicode_data_files(
    llvm_project_root_path: str,
    files=["UnicodeData.txt"],
    base_url="https://www.unicode.org/Public/UCD/latest/ucd/",
) -> None:
    """Fetches necessary unicode data files from unicode.org"""

    for file, url in zip(files, map(lambda file: base_url + file, files)):
        location = f"{llvm_project_root_path}/libc/utils/wctype_utils/data/{file}"
        urlretrieve(url, location)
        print(f"Downloaded {url} in {location}")

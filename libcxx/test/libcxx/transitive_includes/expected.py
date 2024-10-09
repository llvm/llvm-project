#!/usr/bin/env python
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Extract the list of expected transitive includes for the given Standard and header.""",
    )
    parser.add_argument(
        "standard",
        default=None,
        choices=["cxx03", "cxx11", "cxx14", "cxx17", "cxx20", "cxx23", "cxx26"],
    )
    parser.add_argument(
        "header",
        default=None,
        help="The header to extract the expected transitive includes for."
    )
    args = parser.parse_args()

    CSV_ROOT = os.path.dirname(__file__)
    filename = os.path.join(CSV_ROOT, f"{args.standard}.csv")
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.startswith(args.header + ' '):
                print(line, end='') # lines already end in newline

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
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Diff two files.""",
    )
    parser.add_argument("file1", default=None)
    parser.add_argument("file2", default=None)
    args = parser.parse_args()

    def doread(f):
        with open(f, 'r') as file:
            content = file.read()
        lines = [l.strip() for l in content.splitlines()]
        return list(filter(None, lines))

    content1 = doread(args.file1)
    content2 = doread(args.file2)

    for l1, l2 in zip(content1, content2):
        if l1 != l2:
            print("line not equal")
            print(l1)
            print(l2)
            sys.exit(1)

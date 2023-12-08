#!/usr/bin/env python
# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""DExTer entry point. This is the only non-module file."""

import sys

if sys.version_info < (3, 6, 0):
    sys.stderr.write("You need python 3.6 or later to run DExTer\n")
    # Equivalent to sys.exit(ReturnCode._ERROR).
    sys.exit(1)

from dex.tools import main

if __name__ == "__main__":
    return_code = main()
    sys.exit(return_code.value)

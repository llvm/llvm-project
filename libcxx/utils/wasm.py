#!/usr/bin/env python3
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

"""wasm.py is a utility for running a WebAssembly program under a WASI runtime.

It forwards command line arguments and environment variables to the program and
returns the program's error code.
"""

import argparse
import os
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=str, required=True)
    parser.add_argument("--execdir", type=str, required=True)
    parser.add_argument("--env", type=str, action="append", default=[])
    parser.add_argument("--dir", dest="dirs", type=str, action="append", default=[])
    parser.add_argument("test_binary")
    parser.add_argument("test_args", nargs=argparse.ZERO_OR_MORE, default=[])
    args = parser.parse_args()

    if not shutil.which(args.runtime):
        sys.exit(
            f"Failed to find a WASI runtime from --runtime value: '{args.runtime}'"
        )

    if not os.path.exists(args.test_binary):
        sys.exit(f"Expected argument to be a test executable: '{args.test_binary}'")

    # WASI resolves a path against the preopened directory whose name it starts
    # with, so tests need the execution directory granted both as '.' for the
    # relative paths they use and under its absolute name for %t paths.
    execdir = os.path.abspath(args.execdir)
    commandline = [args.runtime, "run", "--dir", ".", "--dir", execdir]
    for directory in args.dirs:
        commandline += ["--dir", directory]
    for env in args.env:
        commandline += ["--env", env]
    commandline += [args.test_binary, *args.test_args]

    return subprocess.call(commandline, cwd=execdir)


if __name__ == "__main__":
    exit(main())

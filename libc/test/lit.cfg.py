# -*- Python -*-
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##
#
# This is the lit configuration for LLVM libc tests.
#
# ===----------------------------------------------------------------------===##

import os
import site
import sys

import lit.formats
import lit.util

# Add libc's utils directory to the path so we can import the test format
site.addsitedir(os.path.join(config.libc_src_root, "utils"))
import LibcTestFormat

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "libc"

# testFormat: Use libc's custom test format that discovers pre-built
# test executables (Libc*Tests) in the build directory.
config.test_format = LibcTestFormat.LibcTest()

# suffixes: Not used by LibcTest format, but kept for compatibility
config.suffixes = []

# excludes: A list of directories to exclude from the testsuite.
config.excludes = ["Inputs", "CMakeLists.txt", "README.txt", "LICENSE.txt", "UnitTest"]

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where test executables are built.
# Set both to the build directory so ExecutableTest finds executables correctly.
config.test_exec_root = os.path.join(config.libc_obj_root, "test")
config.test_source_root = config.test_exec_root

# Add tool directories to PATH (in case we add FileCheck tests later)
if hasattr(config, "llvm_tools_dir") and config.llvm_tools_dir:
    config.environment["PATH"] = os.path.pathsep.join(
        [config.llvm_tools_dir, config.environment.get("PATH", "")]
    )

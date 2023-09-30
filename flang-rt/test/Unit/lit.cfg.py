# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import platform
import re
import subprocess
import sys

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# name: The name of this test suite.
config.name = "flang-rt-Unit"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []

# test_source_root: The root path where unit test binaries are located.
# test_exec_root: The root path where tests should be run.
config.test_source_root = os.path.join(config.flang_rt_obj_root, "unittests")
config.test_exec_root = config.test_source_root

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, "Tests")

# Tweak the PATH to include the flang bin and libs dirs.
path = os.path.pathsep.join(
    (
        config.flang_bin_dir,
        config.llvm_tools_dir,
        config.environment["PATH"]
    )
)
config.environment["PATH"] = path

path = os.path.pathsep.join(
    (
        config.flang_rt_lib_dir,
        config.flang_libs_dir,
        config.flang_bin_dir,
        config.environment.get("LD_LIBRARY_PATH", ""),
    )
)
config.environment["LD_LIBRARY_PATH"] = path

# Propagate PYTHON_EXECUTABLE into the environment
# config.environment['PYTHON_EXECUTABLE'] = sys.executable

# To modify the default target triple for flang-rt tests.
if config.flang_rt_test_triple:
    config.target_triple = config.flang_rt_test_triple
    config.environment[config.llvm_target_triple_env] = config.flang_rt_test_triple

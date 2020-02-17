# -*- Python -*-

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

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'Flang'

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)


# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.f', '.F', '.ff','.FOR', '.for', '.f77', '.f90', '.F90',
                   '.ff90', '.f95', '.F95', '.ff95', '.fpp', '.FPP', '.cuf',
                   '.CUF', '.f18', '.F18']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)


# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.flang_obj_root, 'test-lit')

config.substitutions.append(('%PATH%', config.environment['PATH']))

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.flang_obj_root, 'test-lit')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.flang_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

# For each occurrence of a flang tool name, replace it with the full path to
# the build directory holding that tool.  We explicitly specify the directories
# to search to ensure that we get the tools just built and not some random
# tools that might happen to be in the user's PATH.
tool_dirs = [config.llvm_tools_dir, config.flang_tools_dir]
flang_includes = "-I" + config.flang_intrinsic_modules_dir

tools = [ToolSubst('%flang', command=FindTool('flang'), unresolved='fatal'),
         ToolSubst('%f18', command=FindTool('f18'), unresolved='fatal'),
         ToolSubst('%f18_with_includes', command=FindTool('f18'),
         extra_args=[flang_includes], unresolved='fatal')]

llvm_config.add_tool_substitutions(tools, tool_dirs)


# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'debuginfo-tests'

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.c', '.cpp', '.m']

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.join(config.debuginfo_tests_src_root, 'tests')

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.debuginfo_tests_obj_root

llvm_config.use_default_substitutions()

llvm_config.use_clang()

if config.llvm_use_sanitizer:
    # Propagate path to symbolizer for ASan/MSan.
    llvm_config.with_system_environment(
        ['ASAN_SYMBOLIZER_PATH', 'MSAN_SYMBOLIZER_PATH'])

tool_dirs = [config.llvm_tools_dir]

tools = [
    ToolSubst('%test_debuginfo', command=os.path.join(
        config.llvm_src_root, 'utils', 'test_debuginfo.pl')),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

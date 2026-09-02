# -*- Python -*-

import os
import shlex

import lit.formats

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "Flang Tools"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [
    ".f",
    ".f90",
    ".f95",
    ".f03",
    ".f08",
    ".mod",
    ".test"
]

# Exclude 'Inputs' directories as they are for test dependencies.
config.excludes = ["Inputs"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.flang_tools_binary_dir, "test")

# Tools need the same environment setup as clang (we don't need clang itself).
llvm_config.use_clang(required=False)

python_exec = shlex.quote(config.python_executable)
check_flang_tidy = os.path.join(
    config.test_source_root, "flang-tidy", "check_flang_tidy.py"
)
config.substitutions.append(
    ("%check_flang_tidy", "%s %s" % (python_exec, check_flang_tidy))
)

# Plugins (loadable modules)
if config.has_plugins and config.llvm_plugin_ext:
    config.available_features.add("plugins")

# Disable default configuration files to avoid interference with test runs.
config.environment["FLANG_NO_DEFAULT_CONFIG"] = "1"
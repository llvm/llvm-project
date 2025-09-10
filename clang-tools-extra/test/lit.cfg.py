# -*- Python -*-

import os
import shlex

import lit.formats

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "Clang Tools"

# testFormat: The test format to use to interpret tests.
# We prefer the lit internal shell which provides a better user experience on
# failures and is faster unless the user explicitly disables it with
# LIT_USE_INTERNAL_SHELL=0 env var.
use_lit_shell = True
lit_shell_env = os.environ.get("LIT_USE_INTERNAL_SHELL")
if lit_shell_env:
    use_lit_shell = lit.util.pythonize_bool(lit_shell_env)

config.test_format = lit.formats.ShTest(not use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [
    ".c",
    ".cpp",
    ".hpp",
    ".m",
    ".mm",
    ".cu",
    ".ll",
    ".cl",
    ".s",
    ".modularize",
    ".module-map-checker",
    ".test",
]

# Test-time dependencies located in directories called 'Inputs' are excluded
# from test suites; there won't be any lit tests within them.
config.excludes = ["Inputs"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.clang_tools_binary_dir, "test")

# Tools need the same environment setup as clang (we don't need clang itself).
llvm_config.clang_setup()

if config.clang_tidy_staticanalyzer:
    config.available_features.add("static-analyzer")

python_exec = shlex.quote(config.python_executable)
check_clang_tidy = os.path.join(
    config.test_source_root, "clang-tidy", "check_clang_tidy.py"
)
config.substitutions.append(
    ("%check_clang_tidy", "%s %s" % (python_exec, check_clang_tidy))
)
clang_tidy_diff = os.path.join(
    config.test_source_root, "..", "clang-tidy", "tool", "clang-tidy-diff.py"
)
config.substitutions.append(
    ("%clang_tidy_diff", "%s %s" % (python_exec, clang_tidy_diff))
)
run_clang_tidy = os.path.join(
    config.test_source_root, "..", "clang-tidy", "tool", "run-clang-tidy.py"
)
config.substitutions.append(
    ("%run_clang_tidy", "%s %s" % (python_exec, run_clang_tidy))
)
clang_tidy_headers = os.path.join(
    config.test_source_root, "clang-tidy", "checkers", "Inputs", "Headers"
)
config.substitutions.append(("%clang_tidy_headers", clang_tidy_headers))

# Plugins (loadable modules)
if config.has_plugins and config.llvm_plugin_ext:
    config.available_features.add("plugins")

# It is not realistically possible to account for all options that could
# possibly be present in system and user configuration files, so disable
# default configs for the test runs.
config.environment["CLANG_NO_DEFAULT_CONFIG"] = "1"

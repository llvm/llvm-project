import os

import lit.formats
import lit.util

from lit.llvm import llvm_config
import site

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "libclc"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [
    ".cl",
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.join(os.path.dirname(__file__))

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.test_run_dir, "test")

llvm_config.use_default_substitutions()

llvm_config.use_clang()

tools = []
tool_dirs = [config.llvm_tools_dir]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# TODO: Consolidate the logic for turning on the internal shell by default for all LLVM test suites.
# See https://github.com/llvm/llvm-project/issues/106636 for more details.
#
# We prefer the lit internal shell which provides a better user experience on failures
# unless the user explicitly disables it with LIT_USE_INTERNAL_SHELL=0 env var.
use_lit_shell = True
lit_shell_env = os.environ.get("LIT_USE_INTERNAL_SHELL")
if lit_shell_env:
    use_lit_shell = lit.util.pythonize_bool(lit_shell_env)

config.test_format = lit.formats.ShTest(execute_external=not use_lit_shell)

import os

import lit.llvm

lit.llvm.initialize(lit_config, config)
lit.llvm.llvm_config.use_default_substitutions()

# TODO: Consolidate the logic for turning on the internal shell by default for all LLVM test suites.
# See https://github.com/llvm/llvm-project/issues/106636 for more details.
#
# We prefer the lit internal shell which provides a better user experience on failures
# and is faster unless the user explicitly disables it with LIT_USE_INTERNAL_SHELL=0
# env var.
use_lit_shell = True
lit_shell_env = os.environ.get("LIT_USE_INTERNAL_SHELL")
if lit_shell_env:
    use_lit_shell = lit.util.pythonize_bool(lit_shell_env)

config.name = "ClangIncludeCleaner"
config.suffixes = [".test", ".c", ".cpp"]
config.excludes = ["Inputs"]
config.test_format = lit.formats.ShTest(not use_lit_shell)
config.test_source_root = config.clang_include_cleaner_source_dir + "/test"
config.test_exec_root = config.clang_include_cleaner_binary_dir + "/test"

config.environment["PATH"] = os.path.pathsep.join(
    (config.clang_tools_dir, config.llvm_tools_dir, config.environment["PATH"])
)

# It is not realistically possible to account for all options that could
# possibly be present in system and user configuration files, so disable
# default configs for the test runs.
config.environment["CLANG_NO_DEFAULT_CONFIG"] = "1"

import os

import lit.llvm
import lit.util

lit.llvm.initialize(lit_config, config)
lit.llvm.llvm_config.clang_setup()
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

config.name = "Clangd"
config.suffixes = [".test"]
config.excludes = ["Inputs"]
config.test_format = lit.formats.ShTest(not use_lit_shell)
config.test_source_root = config.clangd_source_dir + "/test"
config.test_exec_root = config.clangd_binary_dir + "/test"


# Used to enable tests based on the required targets. Can be queried with e.g.
#    REQUIRES: x86-registered-target
def calculate_arch_features(arch_string):
    return [arch.lower() + "-registered-target" for arch in arch_string.split()]


lit.llvm.llvm_config.feature_config([("--targets-built", calculate_arch_features)])

# Clangd-specific lit environment.
config.substitutions.append(
    ("%clangd-benchmark-dir", config.clangd_binary_dir + "/benchmarks")
)

if config.clangd_build_xpc:
    config.available_features.add("clangd-xpc-support")

if config.clangd_enable_remote:
    config.available_features.add("clangd-remote-index")

if config.clangd_tidy_checks:
    config.available_features.add("clangd-tidy-checks")

if config.have_zlib:
    config.available_features.add("zlib")

if lit.util.pythonize_bool(config.have_benchmarks):
    config.available_features.add("have-benchmarks")

# It is not realistically possible to account for all options that could
# possibly be present in system and user configuration files, so disable
# default configs for the test runs.
config.environment["CLANG_NO_DEFAULT_CONFIG"] = "1"

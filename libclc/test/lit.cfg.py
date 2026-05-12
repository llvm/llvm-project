"""
Lit configuration file for libclc tests.
"""

import os

import lit.formats

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "libclc"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".cl", ".test"]

# Exclude certain directories from test discovery
config.excludes = ["CMakeLists.txt"]

# test_source_root: The root path where tests are located.
# For per-target tests, this is the target's test directory.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.libclc_obj_root

config.target_triple = config.libclc_target

arch = config.libclc_arch.upper()
config.available_features.add(arch)

llvm_config.use_default_substitutions()

llvm_config.use_clang()

llvm_config.add_tool_substitutions(["llvm-nm"], config.llvm_tools_dir)

config.substitutions.extend([
    ("%libclc_library_dir", config.libclc_library_dir),
    ("%libclc_target", config.libclc_target),
    ("%check_prefix", arch)
])

# Propagate PATH from environment
if "PATH" in os.environ:
    config.environment["PATH"] = os.path.pathsep.join(
        [config.llvm_tools_dir, os.environ["PATH"]]
    )
else:
    config.environment["PATH"] = config.llvm_tools_dir

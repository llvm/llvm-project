"""
Lit configuration file for libclc tests.
"""

import os

import lit.formats

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "libclc"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(True)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".test"]

# Exclude certain directories from test discovery
config.excludes = ["CMakeLists.txt"]

# test_source_root: The root path where tests are located.
# For per-target tests, this is the target's test directory.
config.test_source_root = config.libclc_obj_root

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.libclc_obj_root

# Propagate PATH from environment
if "PATH" in os.environ:
    config.environment["PATH"] = os.path.pathsep.join(
        [config.llvm_tools_dir, os.environ["PATH"]]
    )
else:
    config.environment["PATH"] = config.llvm_tools_dir

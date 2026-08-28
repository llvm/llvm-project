# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os

import lit.formats

# name: The name of this test suite.
config.name = "SYCL-Unit"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []


def prepend_executable_path(path):
    old_path = config.environment.get("PATH")
    config.environment["PATH"] = (
        f"{path}{os.path.pathsep}{old_path}" if old_path else path
    )


# Windows doesn't have rpath so make sure the runtime deps for the unittest
# executables, such as the SYCL runtime library and LLVMOffload.dll, are found.
# The DLLs are placed in the compiler bin dir and the library dir.
if config.operating_system == "Windows":
    if config.bin_dir:
        prepend_executable_path(config.bin_dir)
    if config.library_dir:
        prepend_executable_path(config.library_dir)

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.unittest_dir
config.test_source_root = config.test_exec_root

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, ".unittests")

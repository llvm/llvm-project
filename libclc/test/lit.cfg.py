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

# Exclude certain directories and files from test discovery
config.excludes = [
    "CMakeLists.txt",
    "update_libclc_tests.py",
]

# test_source_root: The root path where tests are located.
# For per-target tests, this is the target's test directory.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.libclc_obj_root

config.target_triple = config.libclc_target

supported_test_architectures = ["amdgcn", "amdgpu"]

config.targets = set()


def calculate_arch_features(arch_string):
    features = []
    for arch in arch_string.split():
        if (
            arch.lower() in supported_test_architectures
            and config.libclc_target_arch.lower() in supported_test_architectures
        ):
            features.append(arch.lower() + "-registered-target")
            config.targets.add(arch.upper())
    return features


llvm_config.feature_config([("--targets-built", calculate_arch_features)])

llvm_config.use_default_substitutions()

llvm_config.use_clang()

llvm_config.add_tool_substitutions(["llvm-nm"], config.llvm_tools_dir)

config.substitutions.extend(
    [
        ("%library_dir", config.libclc_library_dir),
        ("%target", config.libclc_target),
        ("%cpu", config.libclc_target_cpu),
        ("%check_prefix", config.libclc_target_arch.upper()),
    ]
)

# Propagate PATH from environment
if "PATH" in os.environ:
    config.environment["PATH"] = os.path.pathsep.join(
        [config.llvm_tools_dir, os.environ["PATH"]]
    )
else:
    config.environment["PATH"] = config.llvm_tools_dir

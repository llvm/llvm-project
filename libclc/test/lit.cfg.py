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

target_arch = config.libclc_target_arch.lower()
check_prefix = "AMDGCN" if target_arch == "amdgpu" else target_arch.upper()

is_standalone = config.libclc_standalone_build.lower() == "true"
path = os.path.join(config.libclc_library_dir, config.libclc_target, "libclc.bc")
libclc_lib = f"--libclc-lib=:{path}" if is_standalone else ""

config.substitutions.extend(
    [
        ("%library_dir", config.libclc_library_dir),
        ("%target", config.libclc_target),
        ("%cpu", config.libclc_target_cpu),
        ("%libclc_lib", libclc_lib),
        ("%check_prefix", check_prefix),
    ]
)

test_arch = getattr(config, "libclc_test_arch", "")
offload_libdir = getattr(config, "libclc_offload_libdir", "")

if test_arch and offload_libdir and os.path.isfile(path):
    compile_cmd = (
        f"{os.path.join(config.llvm_tools_dir, 'clang')} --target={config.libclc_target} "
        f"-march={test_arch} -cl-std=CL3.0 -nogpulib "
        f"--libclc-lib=:{path} %s -o %t"
    )
    run_cmd = f"{os.path.join(config.llvm_tools_dir, 'llvm-gpu-loader')} --kernel test"
    config.environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [offload_libdir, config.environment.get("LD_LIBRARY_PATH", "")]
    )
    config.substitutions.append(
        ("%libclc-compile-and-run", f"{compile_cmd} && {run_cmd}")
    )
    config.substitutions.append(("%libclc-compile", compile_cmd))
    config.substitutions.append(("%libclc-run", run_cmd))
    config.available_features.add("libclc-native-run")

# Propagate PATH from environment
if "PATH" in os.environ:
    config.environment["PATH"] = os.path.pathsep.join(
        [config.llvm_tools_dir, os.environ["PATH"]]
    )
else:
    config.environment["PATH"] = config.llvm_tools_dir

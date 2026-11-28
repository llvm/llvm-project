# -*- Python -*-

import shlex
import platform
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst, FindTool


def shjoin(args, sep=" "):
    return sep.join([shlex.quote(arg) for arg in args])


# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "flang-rt"

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [
    ".c",
    ".cpp",
    ".f",
    ".F",
    ".ff",
    ".FOR",
    ".for",
    ".f77",
    ".f90",
    ".F90",
    ".ff90",
    ".f95",
    ".F95",
    ".ff95",
    ".fpp",
    ".FPP",
    ".cuf",
    ".CUF",
    ".f18",
    ".F18",
    ".f03",
    ".F03",
    ".f08",
    ".F08",
    ".ll",
    ".fir",
    ".mlir",
]

llvm_config.use_default_substitutions()

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
# lit writes a '.lit_test_times.txt' file into this directory.
config.test_exec_root = config.flang_rt_binary_test_dir

# On MacOS, some tests need -isysroot to build binaries.
isysroot_flag = []
if config.osx_sysroot:
    isysroot_flag = ["-isysroot", config.osx_sysroot]
config.substitutions.append(("%isysroot", " ".join(isysroot_flag)))

flang_args = []
if not config.llvm_tree_available:
    flang_args.append(
        f"-fintrinsic-modules-path={config.flang_rt_output_resource_mod_dir}"
    )

tools = [
    ToolSubst(
        "%flang",
        command=config.flang,
        extra_args=flang_args,
        unresolved="fatal",
    ),
    ToolSubst(
        "%clang",
        command=FindTool("clang"),
        unresolved="fatal",
    ),
    ToolSubst("%cc", command=config.cc, unresolved="fatal"),
]
llvm_config.add_tool_substitutions(tools)

# Let tests find LLVM's standard tools (FileCheck, split-file, not, ...)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

# Include path for C headers that define Flang's Fortran ABI.
config.substitutions.append(
    ("%include", os.path.join(config.flang_source_dir, "include"))
)

# Library path of libflang_rt.runtime.a/.so (for lib search path when using non-Flang driver for linking and LD_LIBRARY_PATH)
config.substitutions.append(("%libdir", config.flang_rt_output_resource_lib_dir))


# TODO: Consolidate this version and the one in compiler-rt into
# llvm/utils/lit/lit/llvm/config.py
def push_dynamic_library_lookup_path(config, new_path):
    if platform.system() == "Windows":
        dynamic_library_lookup_var = "PATH"
    elif platform.system() == "Darwin":
        dynamic_library_lookup_var = "DYLD_LIBRARY_PATH"
    elif platform.system() == "Haiku":
        dynamic_library_lookup_var = "LIBRARY_PATH"
    else:
        dynamic_library_lookup_var = "LD_LIBRARY_PATH"

    new_ld_library_path = os.path.pathsep.join(
        (new_path, config.environment.get(dynamic_library_lookup_var, ""))
    )

    if platform.system() == "Darwin":
        # Workaround an issue in LD which does not use the correct libLTO
        # if the DYLD_LIBRARY_PATH is not normalized.
        new_ld_library_path = os.path.normpath(new_ld_library_path)

    config.environment[dynamic_library_lookup_var] = new_ld_library_path


# Set dynamic library lookup path so that dynamically linked runtimes can be found.
push_dynamic_library_lookup_path(config, config.flang_rt_output_resource_lib_dir)

# For CUDA offloading, additional steps (device linking) and libraries (cudart) are needed.
if config.flang_rt_experimental_offload_support == "CUDA":
    config.available_features.add("offload-cuda")

if config.flang_rt_fortran_modules:
    config.available_features.add("fortran-modules")

# Set OBJECT_MODE=64 as tools on AIX default to 32-bit.
if "system-aix" in config.available_features:
    config.environment["OBJECT_MODE"] = "64"

# Detect Windows Subsystem for Linux (WSL)
uname_r = platform.uname().release
if uname_r.endswith("-Microsoft"):
    config.available_features.add("wsl1")
elif uname_r.endswith("microsoft-standard-WSL2"):
    config.available_features.add("wsl2")

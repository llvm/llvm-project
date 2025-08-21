# -*- Python -*-

import shlex
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
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

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

tools = [
    ToolSubst(
        "%flang",
        command=config.flang,
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

# For CUDA offloading, additional steps (device linking) and libraries (cudart) are needed.
if config.flang_rt_experimental_offload_support == "CUDA":
    config.available_features.add("offload-cuda")

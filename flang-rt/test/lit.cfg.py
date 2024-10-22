# -*- Python -*-

import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst, FindTool

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
config.test_exec_root = config.flangrt_binary_test_dir

# On MacOS, -isysroot is needed to build binaries.
isysroot_flag = []
if config.osx_sysroot:
    isysroot_flag = ["-isysroot", config.osx_sysroot]

tools = [
    ToolSubst(
        "%flang",
        command=FindTool("flang-new"),
        extra_args=isysroot_flag,
        unresolved="fatal",
    )
]

# Define some variables to help us test that the flang runtime doesn't depend on
# the C++ runtime libraries. For this we need a C compiler.
libruntime = os.path.join(config.flangrt_build_lib_dir, "libflang_rt.a")
include = os.path.join(config.flang_source_dir, "include")
tools.append(
    ToolSubst("%cc", command=config.cc, extra_args=isysroot_flag, unresolved="fatal")
)
tools.append(ToolSubst("%libruntime", command=libruntime, unresolved="fatal"))
tools.append(ToolSubst("%include", command=include, unresolved="fatal"))

# Let tests find LLVM's standard tools (FileCheck, split-file, not, ...)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

llvm_config.add_tool_substitutions(tools)

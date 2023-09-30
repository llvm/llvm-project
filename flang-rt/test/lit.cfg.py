# -*- Python -*-

import os
import platform
import re
import subprocess
import sys

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

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

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%pluginext", config.llvm_plugin_ext))

llvm_config.use_default_substitutions()

# ask llvm-config about asserts
llvm_config.feature_config([("--assertion-mode", {"ON": "asserts"})])

# Targets
config.targets = frozenset(config.targets_to_build.split())
for arch in config.targets_to_build.split():
    config.available_features.add(arch.lower() + "-registered-target")

# To modify the default target triple for flang-rt tests.
if config.flang_rt_test_triple:
    config.target_triple = config.flang_rt_test_triple
    config.environment[config.llvm_target_triple_env] = config.flang_rt_test_triple

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# Plugins (loadable modules)
if config.has_plugins:
    config.available_features.add("plugins")

if config.linked_bye_extension:
    config.substitutions.append(("%loadbye", ""))
else:
    config.substitutions.append(
        (
            "%loadbye",
            "-fpass-plugin={}/Bye{}".format(
                config.llvm_shlib_dir, config.llvm_plugin_ext
            ),
        )
    )

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.flang_rt_obj_root, "test")

llvm_config.with_environment("PATH", config.flang_bin_dir, append_path=True)
llvm_config.with_environment("PATH", config.flang_rt_obj_root, append_path=True)
llvm_config.with_environment("PATH", config.flang_libs_dir, append_path=True)
llvm_config.with_environment("PATH", config.flang_rt_lib_dir, append_path=True)

# For each occurrence of a flang tool name, replace it with the full path to
# the build directory holding that tool.
tools = [
    ToolSubst("%flang", command=FindTool("flang-new"), unresolved="fatal"),
    ToolSubst(
        "%flang_fc1",
        command=FindTool("flang-new"),
        extra_args=["-fc1"],
        unresolved="fatal",
    ),
]

# Flang has several unimplemented features. TODO messages are used to mark
# and fail if these features are exercised. Some TODOs exit with a non-zero
# exit code, but others abort the execution in assert builds.
# To catch aborts, the `--crash` option for the `not` command has to be used.
tools.append(ToolSubst("%not_todo_cmd", command=FindTool("not"), unresolved="fatal"))
if "asserts" in config.available_features:
    tools.append(
        ToolSubst(
            "%not_todo_abort_cmd",
            command=FindTool("not"),
            extra_args=["--crash"],
            unresolved="fatal",
        )
    )
else:
    tools.append(
        ToolSubst("%not_todo_abort_cmd", command=FindTool("not"), unresolved="fatal")
    )

# Define some variables to help us test that the flang runtime doesn't depend on
# the C++ runtime libraries. For this we need a C compiler. If for some reason
# we don't have one, we can just disable the test.
if config.cc:
    libruntime_static = os.path.join(config.flang_rt_lib_dir, "libflang-rt.a")
    libruntime_shared = os.path.join(config.flang_rt_lib_dir, "libflang-rt.so")
    include = os.path.join(config.flang_src_dir, "include")

    if (
        os.path.isfile(libruntime_static)
        and os.path.isdir(include)
    ):
        config.available_features.add("c-compiler")
        tools.append(ToolSubst("%cc", command=config.cc, unresolved="fatal"))
        tools.append(ToolSubst("%libruntime", command=libruntime_static, unresolved="fatal"))
        tools.append(ToolSubst("%include", command=include, unresolved="fatal"))

    elif (
        os.path.isfile(libruntime_shared)
        and os.path.isdir(include)
    ):
        config.available_features.add("c-compiler")
        tools.append(ToolSubst("%cc", command=config.cc, unresolved="fatal"))
        tools.append(ToolSubst("%libruntime", command=libruntime_shared, unresolved="fatal"))
        tools.append(ToolSubst("%include", command=include, unresolved="fatal"))


# Add all the tools and their substitutions (if applicable). Use the search paths provided for
# finding the tools.
llvm_config.add_tool_substitutions(tools, config.llvm_tools_dir)

# Enable libpgmath testing
result = lit_config.params.get("LIBPGMATH")
if result:
    config.environment["LIBPGMATH"] = True

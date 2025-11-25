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
config.name = "Flang"

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

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(execute_external=not use_lit_shell)

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
config.substitutions.append(("%llvmshlibdir", config.llvm_shlib_dir))
config.substitutions.append(("%pluginext", config.llvm_plugin_ext))

llvm_config.use_default_substitutions()

# ask llvm-config about asserts
llvm_config.feature_config([("--assertion-mode", {"ON": "asserts"})])

# Targets
config.targets = frozenset(config.targets_to_build.split())
for arch in config.targets_to_build.split():
    config.available_features.add(arch.lower() + "-registered-target")

# To modify the default target triple for flang tests.
if config.flang_test_triple:
    config.target_triple = config.flang_test_triple
    config.environment[config.llvm_target_triple_env] = config.flang_test_triple

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# If the flang examples are built, add examples to the config
if config.flang_examples:
    config.available_features.add("examples")

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
config.test_exec_root = os.path.join(config.flang_obj_root, "test")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.flang_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

if config.flang_standalone_build:
    # For builds with FIR, set path for tco and enable related tests
    if config.flang_llvm_tools_dir != "":
        config.available_features.add("fir")
        if config.llvm_tools_dir != config.flang_llvm_tools_dir:
            llvm_config.with_environment(
                "PATH", config.flang_llvm_tools_dir, append_path=True
            )

# On MacOS, some tests need -isysroot to build binaries.
isysroot_flag = []
if config.osx_sysroot:
    isysroot_flag = ["-isysroot", config.osx_sysroot]
config.substitutions.append(("%isysroot", " ".join(isysroot_flag)))

# Check for DEFAULT_SYSROOT, because when it is set -isysroot has no effect.
if config.default_sysroot:
    config.available_features.add("default_sysroot")


flang_exe = lit.util.which("flang", config.flang_llvm_tools_dir)
if not flang_exe:
    lit_config.fatal(f"Could not identify flang executable")

# Intrinsic paths that are added implicitly by the `flang` driver, but have to be added manually when invoking the frontend `flang -fc1`.
flang_driver_search_args = []

# Intrinsic paths that are added to `flang` as well as `flang -fc1`.
flang_extra_search_args = list(config.flang_test_fortran_flags)


def get_resource_module_intrinsic_dir(modfile):
    # Determine the intrinsic module search path that is added by the driver. If
    # skipping the driver using -fc1, we need to append the path manually.
    flang_intrinsics_dir = subprocess.check_output(
        [flang_exe, *config.flang_test_fortran_flags, f"-print-file-name={modfile}"],
        text=True,
    ).strip()
    flang_intrinsics_dir = os.path.dirname(flang_intrinsics_dir)
    return flang_intrinsics_dir or None


intrinsics_mod_path = get_resource_module_intrinsic_dir("__fortran_builtins.mod")
if intrinsics_mod_path:
    flang_driver_search_args += [f"-fintrinsic-modules-path={intrinsics_mod_path}"]

openmp_mod_path = get_resource_module_intrinsic_dir("omp_lib.mod")
if openmp_mod_path and openmp_mod_path != intrinsics_mod_path:
    flang_driver_search_args += [f"-fintrinsic-modules-path={openmp_mod_path}"]


# If intrinsic modules are not available, disable tests unless they are marked as 'module-independent'.
config.available_features.add("module-independent")
if config.flang_test_enable_modules or intrinsics_mod_path:
    config.available_features.add("flangrt-modules")
else:
    lit_config.warning(
        f"Intrinsic modules not in driver default paths: disabling most tests; Use FLANG_TEST_ENABLE_MODULES=ON to force-enable"
    )
    config.limit_to_features.add("module-independent")

# Determine if OpenMP runtime was built (enable OpenMP tests via REQUIRES in test file)
if config.flang_test_enable_openmp or openmp_mod_path:
    config.available_features.add("openmp_runtime")

    # Search path for omp_lib.h with LLVM_ENABLE_RUNTIMES=openmp
    # FIXME: openmp should write this file into the resource directory
    flang_extra_search_args += [
        "-I",
        f"{config.flang_obj_root}/../../runtimes/runtimes-bins/openmp/runtime/src",
    ]
else:
    lit_config.warning(
        f"OpenMP modules found not in driver default paths: OpenMP tests disabled; Use FLANG_TEST_ENABLE_OPENMP=ON to force-enable"
    )


lit_config.note(f"using flang: {flang_exe}")
lit_config.note(
    f"using flang implicit search paths: {' '.join(flang_driver_search_args)}"
)
lit_config.note(f"using flang extra search paths: {' '.join(flang_extra_search_args)}")

# For each occurrence of a flang tool name, replace it with the full path to
# the build directory holding that tool.
tools = [
    ToolSubst(
        "bbc",
        command=FindTool("bbc"),
        extra_args=flang_driver_search_args + flang_extra_search_args,
        unresolved="fatal",
    ),
    ToolSubst(
        "%flang",
        command=flang_exe,
        extra_args=flang_extra_search_args,
        unresolved="fatal",
    ),
    ToolSubst(
        "%flang_fc1",
        command=flang_exe,
        extra_args=["-fc1"] + flang_driver_search_args + flang_extra_search_args,
        unresolved="fatal",
    ),
    # Variant that does not implicitly add intrinsic search paths
    ToolSubst(
        "%bbc_bare",
        command=FindTool("bbc"),
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

# Add all the tools and their substitutions (if applicable). Use the search paths provided for
# finding the tools.
if config.flang_standalone_build:
    llvm_config.add_tool_substitutions(
        tools, [config.flang_llvm_tools_dir, config.llvm_tools_dir]
    )
else:
    llvm_config.add_tool_substitutions(tools, config.llvm_tools_dir)

llvm_config.use_clang(required=False)

# Clang may need the include path for ISO_fortran_binding.h.
config.substitutions.append(("%flang_include", config.flang_headers_dir))

# Enable libpgmath testing
result = lit_config.params.get("LIBPGMATH")
if result:
    config.environment["LIBPGMATH"] = True

config.substitutions.append(("%openmp_flags", "-fopenmp"))

# Add features and substitutions to test F128 math support.
# %f128-lib substitution may be used to generate check prefixes
# for LIT tests checking for F128 library support.
if config.flang_runtime_f128_math_lib or config.have_ldbl_mant_dig_113:
    config.available_features.add("flang-supports-f128-math")
if config.flang_runtime_f128_math_lib:
    config.available_features.add(
        "flang-f128-math-lib-" + config.flang_runtime_f128_math_lib
    )
    config.substitutions.append(
        ("%f128-lib", config.flang_runtime_f128_math_lib.upper())
    )
else:
    config.substitutions.append(("%f128-lib", "NONE"))

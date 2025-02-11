import os
import platform
import re
import subprocess
import sys

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "cross-project-tests"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".c", ".cl", ".cpp", ".m", ".ll", ".test"]

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs"]

# test_source_root: The root path where tests are located.
config.test_source_root = config.cross_project_tests_src_root

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.cross_project_tests_obj_root

llvm_config.use_default_substitutions()

tools = [
    ToolSubst(
        "%test_debuginfo",
        command=os.path.join(
            config.cross_project_tests_src_root,
            "debuginfo-tests",
            "llgdb-tests",
            "test_debuginfo.pl",
        ),
    ),
    ToolSubst("%llvm_src_root", config.llvm_src_root),
    ToolSubst("%llvm_tools_dir", config.llvm_tools_dir),
]


def get_required_attr(config, attr_name):
    attr_value = getattr(config, attr_name, None)
    if attr_value is None:
        lit_config.fatal(
            "No attribute %r in test configuration! You may need to run "
            "tests from your build directory or add this attribute "
            "to lit.site.cfg " % attr_name
        )
    return attr_value


# If this is an MSVC environment, the tests at the root of the tree are
# unsupported. The local win_cdb test suite, however, is supported.
is_msvc = get_required_attr(config, "is_msvc")
if is_msvc:
    config.available_features.add("msvc")
    # FIXME: We should add some llvm lit utility code to find the Windows SDK
    # and set up the environment appopriately.
    win_sdk = "C:/Program Files (x86)/Windows Kits/10/"
    arch = "x64"
    llvm_config.with_system_environment(["LIB", "LIBPATH", "INCLUDE"])
    # Clear _NT_SYMBOL_PATH to prevent cdb from attempting to load symbols from
    # the network.
    llvm_config.with_environment("_NT_SYMBOL_PATH", "")
    tools.append(
        ToolSubst("%cdb", '"%s"' % os.path.join(win_sdk, "Debuggers", arch, "cdb.exe"))
    )

# clang_src_dir and lld_src_dir are not used by these tests, but are required by
# use_clang() and use_lld() respectively, so set them to "", if needed.
if not hasattr(config, "clang_src_dir"):
    config.clang_src_dir = ""
llvm_config.use_clang(required=("clang" in config.llvm_enabled_projects))

if not hasattr(config, "lld_src_dir"):
    config.lld_src_dir = ""
llvm_config.use_lld(required=("lld" in config.llvm_enabled_projects))

if "compiler-rt" in config.llvm_enabled_projects:
    config.available_features.add("compiler-rt")

# Check which debuggers are available:
lldb_path = llvm_config.use_llvm_tool("lldb", search_env="LLDB")

if lldb_path is not None:
    config.available_features.add("lldb")

for tool in ["llvm-ar", "llvm-lto2", "opt"]:
    if llvm_config.use_llvm_tool(tool):
        config.available_features.add(tool)

def configure_dexter_substitutions():
    """Configure substitutions for host platform and return list of dependencies"""
    # Produce dexter path, lldb path, and combine into the %dexter substitution
    # for running a test.
    dexter_path = os.path.join(
        config.cross_project_tests_src_root, "debuginfo-tests", "dexter", "dexter.py"
    )
    dexter_test_cmd = '"{}" "{}" test'.format(sys.executable, dexter_path)
    if lldb_path is not None:
        dexter_test_cmd += ' --lldb-executable "{}"'.format(lldb_path)
    tools.append(ToolSubst("%dexter", dexter_test_cmd))

    # For testing other bits of dexter that aren't under the "test" subcommand,
    # have a %dexter_base substitution.
    dexter_base_cmd = '"{}" "{}"'.format(sys.executable, dexter_path)
    tools.append(ToolSubst("%dexter_base", dexter_base_cmd))

    # Set up commands for DexTer regression tests.
    # Builder, debugger, optimisation level and several other flags differ
    # depending on whether we're running a unix like or windows os.
    if platform.system() == "Windows":
        # The Windows builder script uses lld.
        dependencies = ["clang", "lld-link"]
        dexter_regression_test_builder = "clang-cl"
        dexter_regression_test_debugger = "dbgeng"
        dexter_regression_test_flags = "/Zi /Od"
    else:
        # Use lldb as the debugger on non-Windows platforms.
        dependencies = ["clang", "lldb"]
        dexter_regression_test_builder = "clang++"
        dexter_regression_test_debugger = "lldb"
        dexter_regression_test_flags = "-O0 -glldb -std=gnu++11"

    tools.append(
        ToolSubst("%dexter_regression_test_builder", dexter_regression_test_builder)
    )
    tools.append(
        ToolSubst("%dexter_regression_test_debugger", dexter_regression_test_debugger)
    )
    # We don't need to distinguish cflags and ldflags because for Dexter
    # regression tests we use clang to drive the linker, and so all flags will be
    # passed in a single command.
    tools.append(
        ToolSubst("%dexter_regression_test_flags", dexter_regression_test_flags)
    )

    # Typical command would take the form:
    # ./path_to_py/python.exe ./path_to_dex/dexter.py test --fail-lt 1.0 -w --binary %t --debugger lldb --cflags '-O0 -g'
    dexter_regression_test_run = " ".join(
        # "python", "dexter.py", test, fail_mode, builder, debugger, cflags, ldflags
        [
            '"{}"'.format(sys.executable),
            '"{}"'.format(dexter_path),
            "test",
            "--fail-lt 1.0 -w",
            "--debugger",
            dexter_regression_test_debugger,
        ]
    )
    tools.append(ToolSubst("%dexter_regression_test_run", dexter_regression_test_run))

    # Include build flags for %dexter_regression_test.
    dexter_regression_test_build = " ".join(
        [
            dexter_regression_test_builder,
            dexter_regression_test_flags,
        ]
    )
    tools.append(ToolSubst("%dexter_regression_test_build", dexter_regression_test_build))
    return dependencies


def add_host_triple(clang):
    return "{} --target={}".format(clang, config.host_triple)


# The set of arches we can build.
targets = set(config.targets_to_build)
# Add aliases to the target set.
if "AArch64" in targets:
    targets.add("arm64")
if "ARM" in config.targets_to_build:
    targets.add("thumbv7")


def can_target_host():
    # Check if the targets set contains anything that looks like our host arch.
    # The arch name in the triple and targets set may be spelled differently
    # (e.g. x86 vs X86).
    return any(config.host_triple.lower().startswith(x.lower()) for x in targets)


# Dexter tests run on the host machine. If the host arch is supported add
# 'dexter' as an available feature and force the dexter tests to use the host
# triple.
if can_target_host():
    if config.host_triple != config.target_triple:
        print("Forcing dexter tests to use host triple {}.".format(config.host_triple))
    dependencies = configure_dexter_substitutions()
    if all(d in config.available_features for d in dependencies):
        config.available_features.add("dexter")
        llvm_config.with_environment(
            "PATHTOCLANG", add_host_triple(llvm_config.config.clang)
        )
        llvm_config.with_environment(
            "PATHTOCLANGPP", add_host_triple(llvm_config.use_llvm_tool("clang++"))
        )
        llvm_config.with_environment(
            "PATHTOCLANGCL", add_host_triple(llvm_config.use_llvm_tool("clang-cl"))
        )
else:
    print(
        "Host triple {} not supported. Skipping dexter tests in the "
        "debuginfo-tests project.".format(config.host_triple)
    )

tool_dirs = [config.llvm_tools_dir]

llvm_config.add_tool_substitutions(tools, tool_dirs)

lit.util.usePlatformSdkOnDarwin(config, lit_config)

if platform.system() == "Darwin":
    xcode_lldb_vers = subprocess.check_output(["xcrun", "lldb", "--version"]).decode(
        "utf-8"
    )
    match = re.search(r"lldb-(\d+)", xcode_lldb_vers)
    if match:
        apple_lldb_vers = int(match.group(1))
        if apple_lldb_vers < 1000:
            config.available_features.add("apple-lldb-pre-1000")


def get_gdb_version_string():
    """Return gdb's version string, or None if gdb cannot be found or the
    --version output is formatted unexpectedly.
    """
    # See if we can get a gdb version, e.g.
    #   $ gdb --version
    #   GNU gdb (GDB) 10.2
    #   ...More stuff...
    try:
        gdb_vers_lines = (
            subprocess.check_output(["gdb", "--version"]).decode().splitlines()
        )
    except:
        return None  # We coudln't find gdb or something went wrong running it.
    if len(gdb_vers_lines) < 1:
        print("Unkown GDB version format (too few lines)", file=sys.stderr)
        return None
    match = re.search(r"GNU gdb \(.*?\) ((\d|\.)+)", gdb_vers_lines[0].strip())
    if match is None:
        print(f"Unkown GDB version format: {gdb_vers_lines[0]}", file=sys.stderr)
        return None
    return match.group(1)


def get_clang_default_dwarf_version_string(triple):
    """Return the default dwarf version string for clang on this (host) platform
    or None if we can't work it out.
    """
    # Get the flags passed by the driver and look for -dwarf-version.
    cmd = f'{llvm_config.use_llvm_tool("clang")} -g -xc  -c - -v -### --target={triple}'
    stderr = subprocess.run(cmd.split(), stderr=subprocess.PIPE).stderr.decode()
    match = re.search(r"-dwarf-version=(\d+)", stderr)
    if match is None:
        print("Cannot determine default dwarf version", file=sys.stderr)
        return None
    return match.group(1)


# Some cross-project-tests use gdb, but not all versions of gdb are compatible
# with clang's dwarf. Add feature `gdb-clang-incompatibility` to signal that
# there's an incompatibility between clang's default dwarf version for this
# platform and the installed gdb version.
dwarf_version_string = get_clang_default_dwarf_version_string(config.host_triple)
gdb_version_string = get_gdb_version_string()
if dwarf_version_string and gdb_version_string:
    if int(dwarf_version_string) >= 5:
        try:
            from packaging import version
        except:
            lit_config.fatal("Running gdb tests requires the packaging package")
        if version.parse(gdb_version_string) < version.parse("10.1"):
            # Example for llgdb-tests, which use lldb on darwin but gdb elsewhere:
            # XFAIL: !system-darwin && gdb-clang-incompatibility
            config.available_features.add("gdb-clang-incompatibility")
            print(
                "XFAIL some tests: use gdb version >= 10.1 to restore test coverage",
                file=sys.stderr,
            )

llvm_config.feature_config([("--build-mode", {"Debug|RelWithDebInfo": "debug-info"})])

# Allow 'REQUIRES: XXX-registered-target' in tests.
for arch in config.targets_to_build:
    config.available_features.add(arch.lower() + "-registered-target")

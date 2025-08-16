# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "Clang"

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [
    ".c",
    ".cpp",
    ".i",
    ".cir",
    ".cppm",
    ".m",
    ".mm",
    ".cu",
    ".cuh",
    ".hip",
    ".hlsl",
    ".ll",
    ".cl",
    ".clcpp",
    ".s",
    ".S",
    ".modulemap",
    ".test",
    ".rs",
    ".ifs",
    ".rc",
]

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
    "debuginfo-tests",
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.clang_obj_root, "test")

llvm_config.use_default_substitutions()

llvm_config.use_clang()

config.substitutions.append(("%src_dir", config.clang_src_dir))

config.substitutions.append(("%src_include_dir", config.clang_src_dir + "/include"))

config.substitutions.append(("%target_triple", config.target_triple))

config.substitutions.append(("%PATH%", config.environment["PATH"]))


# For each occurrence of a clang tool name, replace it with the full path to
# the build directory holding that tool.  We explicitly specify the directories
# to search to ensure that we get the tools just built and not some random
# tools that might happen to be in the user's PATH.
tool_dirs = [config.clang_tools_dir, config.llvm_tools_dir]

tools = [
    "apinotes-test",
    "c-index-test",
    "cir-opt",
    "clang-diff",
    "clang-format",
    "clang-repl",
    "clang-offload-packager",
    "clang-tblgen",
    "clang-scan-deps",
    "clang-installapi",
    "opt",
    "llvm-ifs",
    "yaml2obj",
    "clang-linker-wrapper",
    "clang-nvlink-wrapper",
    "clang-sycl-linker",
    "llvm-lto",
    "llvm-lto2",
    "llvm-profdata",
    "llvm-readtapi",
    ToolSubst(
        "%clang_extdef_map",
        command=FindTool("clang-extdef-mapping"),
        unresolved="ignore",
    ),
]

if config.clang_examples:
    config.available_features.add("examples")


def have_host_out_of_process_jit_feature_support():
    clang_repl_exe = lit.util.which("clang-repl", config.clang_tools_dir)

    if not clang_repl_exe:
        return False

    testcode = b"\n".join([b"int i = 0;", b"%quit"])

    try:
        clang_repl_cmd = subprocess.run(
            [clang_repl_exe, "-orc-runtime", "-oop-executor"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            input=testcode,
        )
    except OSError:
        return False

    if clang_repl_cmd.returncode == 0:
        return True

    return False

def have_host_jit_feature_support(feature_name):
    clang_repl_exe = lit.util.which("clang-repl", config.clang_tools_dir)

    if not clang_repl_exe:
        return False

    try:
        clang_repl_cmd = subprocess.Popen(
            [clang_repl_exe, "--host-supports-" + feature_name], stdout=subprocess.PIPE
        )
    except OSError:
        print("could not exec clang-repl")
        return False

    clang_repl_out = clang_repl_cmd.stdout.read().decode("ascii")
    clang_repl_cmd.wait()

    return "true" in clang_repl_out

def have_host_clang_repl_cuda():
    clang_repl_exe = lit.util.which('clang-repl', config.clang_tools_dir)

    if not clang_repl_exe:
        return False

    testcode = b'\n'.join([
        b"__global__ void test_func() {}",
        b"test_func<<<1,1>>>();",
        b"extern \"C\" int puts(const char *s);",
        b"puts(cudaGetLastError() ? \"failure\" : \"success\");",
        b"%quit"
    ])
    try:
        clang_repl_cmd = subprocess.run([clang_repl_exe, '--cuda'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        input=testcode)
    except OSError:
        return False

    if clang_repl_cmd.returncode == 0:
        if clang_repl_cmd.stdout.find(b"success") != -1:
            return True

    return False

if have_host_jit_feature_support('jit'):
    config.available_features.add('host-supports-jit')

    if have_host_clang_repl_cuda():
        config.available_features.add('host-supports-cuda')

    if have_host_out_of_process_jit_feature_support():
        config.available_features.add("host-supports-out-of-process-jit")

if config.clang_staticanalyzer:
    config.available_features.add("staticanalyzer")
    tools.append("clang-check")

    if config.clang_staticanalyzer_z3:
        config.available_features.add("z3")
        if config.clang_staticanalyzer_z3_mock:
            config.available_features.add("z3-mock")
    else:
        config.available_features.add("no-z3")

    check_analyzer_fixit_path = os.path.join(
        config.test_source_root, "Analysis", "check-analyzer-fixit.py"
    )
    config.substitutions.append(
        (
            "%check_analyzer_fixit",
            '"%s" %s' % (config.python_executable, check_analyzer_fixit_path),
        )
    )

    csv2json_path = os.path.join(config.test_source_root, "Analysis", "csv2json.py")
    config.substitutions.append(
        (
            "%csv2json",
            '"%s" %s' % (config.python_executable, csv2json_path),
        )
    )

llvm_config.add_tool_substitutions(tools, tool_dirs)

config.substitutions.append(
    (
        "%hmaptool",
        "'%s' %s"
        % (
            config.python_executable,
            os.path.join(config.clang_src_dir, "utils", "hmaptool", "hmaptool"),
        ),
    )
)

config.substitutions.append(
    (
        "%deps-to-rsp",
        '"%s" %s'
        % (
            config.python_executable,
            os.path.join(config.clang_src_dir, "utils", "module-deps-to-rsp.py"),
        ),
    )
)

# Determine whether the test target is compatible with execution on the host.
if "aarch64" in config.host_arch:
    config.available_features.add("aarch64-host")

# Some tests are sensitive to whether clang is statically or dynamically linked
# to other libraries.
if not (config.build_shared_libs or config.link_llvm_dylib or config.link_clang_dylib):
    config.available_features.add("static-libs")

# Plugins (loadable modules)
if config.has_plugins and config.llvm_plugin_ext:
    config.available_features.add("plugins")

if config.clang_default_pie_on_linux:
    config.available_features.add("default-pie-on-linux")

# Set available features we allow tests to conditionalize on.
#
if config.clang_default_cxx_stdlib != "":
    config.available_features.add(
        "default-cxx-stdlib={}".format(config.clang_default_cxx_stdlib)
    )

# As of 2011.08, crash-recovery tests still do not pass on FreeBSD.
if platform.system() not in ["FreeBSD"]:
    config.available_features.add("crash-recovery")

# ANSI escape sequences in non-dumb terminal
if platform.system() not in ["Windows"]:
    config.available_features.add("ansi-escape-sequences")

# Capability to print utf8 to the terminal.
# Windows expects codepage, unless Wide API.
if platform.system() not in ["Windows"]:
    config.available_features.add("utf8-capable-terminal")

# Support for libgcc runtime. Used to rule out tests that require
# clang to run with -rtlib=libgcc.
if platform.system() not in ["Darwin", "Fuchsia"]:
    config.available_features.add("libgcc")

# Case-insensitive file system


def is_filesystem_case_insensitive():
    os.makedirs(config.test_exec_root, exist_ok=True)
    handle, path = tempfile.mkstemp(prefix="case-test", dir=config.test_exec_root)
    isInsensitive = os.path.exists(
        os.path.join(os.path.dirname(path), os.path.basename(path).upper())
    )
    os.close(handle)
    os.remove(path)
    return isInsensitive


if is_filesystem_case_insensitive():
    config.available_features.add("case-insensitive-filesystem")

# Tests that require the /dev/fd filesystem.
if os.path.exists("/dev/fd/0") and sys.platform not in ["cygwin"]:
    config.available_features.add("dev-fd-fs")

# Set on native MS environment.
if re.match(r".*-(windows-msvc)$", config.target_triple):
    config.available_features.add("ms-sdk")

# [PR8833] LLP64-incompatible tests
if not re.match(
    r"^(aarch64|arm64ec|x86_64).*-(windows-msvc|windows-gnu)$", config.target_triple
):
    config.available_features.add("LP64")

# Tests that are specific to the Apple Silicon macOS.
if re.match(r"^arm64(e)?-apple-(macos|darwin)", config.target_triple):
    config.available_features.add("apple-silicon-mac")

# [PR18856] Depends to remove opened file. On win32, a file could be removed
# only if all handles were closed.
if platform.system() not in ["Windows"]:
    config.available_features.add("can-remove-opened-file")

# Features
known_arches = ["x86_64", "mips64", "ppc64", "aarch64"]
if any(config.target_triple.startswith(x) for x in known_arches):
    config.available_features.add("clang-target-64-bits")


def calculate_arch_features(arch_string):
    features = []
    for arch in arch_string.split():
        features.append(arch.lower() + "-registered-target")
    return features


llvm_config.feature_config(
    [
        ("--assertion-mode", {"ON": "asserts"}),
        ("--cxxflags", {r"-D_GLIBCXX_DEBUG\b": "libstdcxx-safe-mode"}),
        ("--targets-built", calculate_arch_features),
    ]
)

if lit.util.which("xmllint"):
    config.available_features.add("xmllint")

if config.enable_backtrace:
    config.available_features.add("backtrace")

if config.enable_threads:
    config.available_features.add("thread_support")

# Check if we should allow outputs to console.
run_console_tests = int(lit_config.params.get("enable_console", "0"))
if run_console_tests != 0:
    config.available_features.add("console")

lit.util.usePlatformSdkOnDarwin(config, lit_config)
macOSSDKVersion = lit.util.findPlatformSdkVersionOnMacOS(config, lit_config)
if macOSSDKVersion is not None:
    config.available_features.add("macos-sdk-" + str(macOSSDKVersion))

if os.path.exists("/etc/gentoo-release"):
    config.available_features.add("gentoo")

if config.enable_shared:
    config.available_features.add("enable_shared")

# Add a vendor-specific feature.
if config.clang_vendor_uti:
    config.available_features.add("clang-vendor=" + config.clang_vendor_uti)

if config.have_llvm_driver:
    config.available_features.add("llvm-driver")


# Some tests perform deep recursion, which requires a larger pthread stack size
# than the relatively low default of 192 KiB for 64-bit processes on AIX. The
# `AIXTHREAD_STK` environment variable provides a non-intrusive way to request
# a larger pthread stack size for the tests. Various applications and runtime
# libraries on AIX use a default pthread stack size of 4 MiB, so we will use
# that as a default value here.
if "AIXTHREAD_STK" in os.environ:
    config.environment["AIXTHREAD_STK"] = os.environ["AIXTHREAD_STK"]
elif platform.system() == "AIX":
    config.environment["AIXTHREAD_STK"] = "4194304"

# Some tools support an environment variable "OBJECT_MODE" on AIX OS, which
# controls the kind of objects they will support. If there is no "OBJECT_MODE"
# environment variable specified, the default behaviour is to support 32-bit
# objects only. In order to not affect most test cases, which expect to support
# 32-bit and 64-bit objects by default, set the environment variable
# "OBJECT_MODE" to "any" by default on AIX OS.

if "system-aix" in config.available_features:
   config.substitutions.append(("llvm-nm", "env OBJECT_MODE=any llvm-nm"))
   config.substitutions.append(("llvm-ar", "env OBJECT_MODE=any llvm-ar"))
   config.substitutions.append(("llvm-ranlib", "env OBJECT_MODE=any llvm-ranlib"))

# It is not realistically possible to account for all options that could
# possibly be present in system and user configuration files, so disable
# default configs for the test runs.
config.environment["CLANG_NO_DEFAULT_CONFIG"] = "1"

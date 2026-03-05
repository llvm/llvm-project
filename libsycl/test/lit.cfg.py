# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import subprocess

from lit.llvm import llvm_config
import lit.formats
from lit.llvm.subst import ToolSubst, FindTool

# name: The name of this test suite.
config.name = "libsycl"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".cpp"]

config.excludes = ["Inputs"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# allow expanding substitutions that are based on other substitutions
config.recursiveExpansionLimit = 10

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.libsycl_obj_root

# To be filled by lit.local.cfg files.
config.required_features = []
config.unsupported_features = []

# Cleanup environment variables which may affect tests
possibly_dangerous_env_vars = [
    "COMPILER_PATH",
    "RC_DEBUG_OPTIONS",
    "CINDEXTEST_PREAMBLE_FILE",
    "LIBRARY_PATH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "OBJC_INCLUDE_PATH",
    "OBJCPLUS_INCLUDE_PATH",
    "LIBCLANG_TIMING",
    "LIBCLANG_OBJTRACKING",
    "LIBCLANG_LOGGING",
    "LIBCLANG_BGPRIO_INDEX",
    "LIBCLANG_BGPRIO_EDIT",
    "LIBCLANG_NOTHREADS",
    "LIBCLANG_RESOURCE_USAGE",
    "LIBCLANG_CODE_COMPLETION_LOGGING",
    "INCLUDE",
]

for name in possibly_dangerous_env_vars:
    if name in llvm_config.config.environment:
        del llvm_config.config.environment[name]

# Propagate some variables from the host environment.
llvm_config.with_system_environment(
    [
        "PATH",
    ]
)

# Take into account extra system environment variables if provided via parameter.
if config.extra_system_environment:
    lit_config.note(
        "Extra system variables to propagate value from: "
        + config.extra_system_environment
    )
    extra_env_vars = config.extra_system_environment.split(",")
    for var in extra_env_vars:
        if var in os.environ:
            llvm_config.with_system_environment(var)

llvm_config.with_environment("PATH", config.lit_tools_dir, append_path=True)

# Configure LD_LIBRARY_PATH
llvm_config.with_system_environment(
    ["LD_LIBRARY_PATH", "LIBRARY_PATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"]
)
llvm_config.with_environment(
    "LD_LIBRARY_PATH", config.libsycl_libs_dir, append_path=True
)

llvm_config.with_environment("PATH", config.libsycl_tools_dir, append_path=True)

if config.extra_environment:
    lit_config.note("Extra environment variables")
    for env_pair in config.extra_environment.split(","):
        [var, val] = env_pair.split("=", 1)
        if val:
            llvm_config.with_environment(var, val)
            lit_config.note("\t" + var + "=" + val)
        else:
            lit_config.note("\tUnset " + var)
            llvm_config.with_environment(var, "")


# Temporarily modify environment to be the same that we use when running tests
class test_env:
    def __enter__(self):
        self.old_environ = dict(os.environ)
        os.environ.clear()
        os.environ.update(config.environment)
        self.old_dir = os.getcwd()
        os.chdir(config.libsycl_obj_root)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        os.environ.clear()
        os.environ.update(self.old_environ)
        os.chdir(self.old_dir)


# General substritutions
config.substitutions.append(
    (
        "%sycl_options",
        " -lsycl"
        + " -isystem "
        + config.libsycl_include
        + " -isystem "
        + os.path.join(config.libsycl_include, "sycl")
        + " -L"
        + config.libsycl_libs_dir,
    )
)
config.substitutions.append(("%sycl_libs_dir", config.libsycl_libs_dir))
config.substitutions.append(("%sycl_static_libs_dir", config.libsycl_libs_dir))
config.substitutions.append(("%obj_ext", ".o"))
config.substitutions.append(("%sycl_include", "-isystem " + config.libsycl_include))
config.substitutions.append(("%include_option", "-include"))
config.substitutions.append(("%debug_option", "-g"))
config.substitutions.append(("%cxx_std_option", "-std="))
config.substitutions.append(("%fPIC", "-fPIC"))
config.substitutions.append(("%shared_lib", "-shared"))
config.substitutions.append(("%O0", "-O0"))

sycl_ls = FindTool("sycl-ls").resolve(
    llvm_config, os.pathsep.join([config.libsycl_bin_dir, config.llvm_tools_dir])
)
if not sycl_ls:
    lit_config.fatal("can't find `sycl-ls`")

tools = [
    ToolSubst("FileCheck", unresolved="ignore"),
    # not is only substituted in certain circumstances; this is lit's default
    # behaviour.
    ToolSubst(
        r"\| \bnot\b", command=FindTool("not"), verbatim=True, unresolved="ignore"
    ),
    ToolSubst("sycl-ls", command=sycl_ls, unresolved="fatal"),
]

# Try and find each of these tools in the libsycl bin directory, in the llvm tools directory
# or the PATH, in that order. If found, they will be added as substitutions with the full path
# to the tool.
llvm_config.add_tool_substitutions(
    tools, [config.libsycl_bin_dir, config.llvm_tools_dir, os.environ.get("PATH", "")]
)

lit_config.note("Targeted devices: all")
with test_env():
    sycl_ls_output = subprocess.check_output(sycl_ls, text=True, shell=True)

    devices = set()
    for line in sycl_ls_output.splitlines():
        if not line.startswith("["):
            continue
        backend, device = line[1:].split("]")[0].split(":")
        devices.add("{}:{}".format(backend, device))
    libsycl_devices = list(devices)

available_devices = {
    "level_zero": "gpu",
}
for d in libsycl_devices:
    be, dev = d.split(":")
    if be not in available_devices:
        lit_config.error("Unsupported device {}".format(d))
    if dev not in available_devices[be]:
        lit_config.error("Unsupported device {}".format(d))

if len(libsycl_devices) > 0:
    config.available_features.add("any-device")

for sycl_device in libsycl_devices:
    be, dev = sycl_device.split(":")
    config.available_features.add("any-device-is-" + dev)
    config.available_features.add("any-device-is-" + be)

# Check if user passed verbose-print parameter, if yes, add VERBOSE_PRINT macro
if "verbose-print" in lit_config.params:
    verbose_print = "-DVERBOSE_PRINT"
else:
    verbose_print = ""

clangxx = " " + config.libsycl_compiler + " -Werror " + config.cxx_flags + verbose_print
config.substitutions.append(("%clangxx", clangxx))

config.test_format = lit.formats.ShTest()

try:
    import psutil

    # Set timeout for a single test
    lit_config.maxIndividualTestTime = 600

except ImportError:
    pass

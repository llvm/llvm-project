# -*- Python -*-

import os
import platform

import lit.formats

# name: The name of this test suite.
config.name = "flang-rt-OldUnit"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".test", ".test.exe"]

# test_source_root: The root path where unit test binaries are located.
config.test_source_root = os.path.join(config.flangrt_binary_dir, "unittests")

# test_exec_root: The root path where tests should be run.
# lit writes a '.lit_test_times.txt' file into this directory.
config.test_exec_root = config.flang_rt_binary_test_dir

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ExecutableTest()


def find_shlibpath_var():
    if platform.system() in ["Linux", "FreeBSD", "NetBSD", "OpenBSD", "SunOS"]:
        yield "LD_LIBRARY_PATH"
    elif platform.system() == "Darwin":
        yield "DYLD_LIBRARY_PATH"
    elif platform.system() == "Windows" or sys.platform == "cygwin":
        yield "PATH"
    elif platform.system() == "AIX":
        yield "LIBPATH"


for shlibpath_var in find_shlibpath_var():
    config.environment[shlibpath_var] = os.path.pathsep.join(
        (
            config.flang_rt_output_resource_lib_dir,
            config.environment.get(shlibpath_var, ""),
        )
    )

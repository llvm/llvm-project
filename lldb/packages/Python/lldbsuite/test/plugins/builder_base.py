"""
If the build* function is passed the compiler argument, for example, 'llvm-gcc',
it is passed as a make variable to the make command.  Otherwise, we check the
LLDB_CC environment variable; if it is defined, it is passed as a make variable
to the make command.

If neither the compiler keyword argument nor the LLDB_CC environment variable is
specified, no CC make variable is passed to the make command.  The Makefile gets
to define the default CC being used.

Same idea holds for LLDB_ARCH environment variable, which maps to the ARCH make
variable.
"""

# System imports
import os
import platform
import subprocess
import sys

# Our imports
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test import configuration
from lldbsuite.test_event import build_exception


def getArchitecture():
    """Returns the architecture in effect the test suite is running with."""
    return configuration.arch if configuration.arch else ""


def getCompiler():
    """Returns the compiler in effect the test suite is running with."""
    compiler = configuration.compiler if configuration.compiler else "clang"
    compiler = lldbutil.which(compiler)
    return os.path.realpath(compiler)


def getArchFlag():
    """Returns the flag required to specify the arch"""
    compiler = getCompiler()
    if compiler is None:
        return ""
    elif "gcc" in compiler:
        archflag = "-m"
    elif "clang" in compiler:
        archflag = "-arch"
    else:
        archflag = None

    return ("ARCHFLAG=" + archflag) if archflag else ""

def getMake(test_subdir, test_name):
    """Returns the invocation for GNU make.
       The first argument is a tuple of the relative path to the testcase
       and its filename stem."""
    if platform.system() == "FreeBSD" or platform.system() == "NetBSD":
        make = "gmake"
    else:
        make = "make"

    # Construct the base make invocation.
    lldb_test = os.environ["LLDB_TEST"]
    lldb_test_src = os.environ["LLDB_TEST_SRC"]
    lldb_build = os.environ["LLDB_BUILD"]
    if not (lldb_test and lldb_test_src and lldb_build and test_subdir and
            test_name and (not os.path.isabs(test_subdir))):
        raise Exception("Could not derive test directories")
    build_dir = os.path.join(lldb_build, test_subdir, test_name)
    src_dir = os.path.join(lldb_test_src, test_subdir)
    # This is a bit of a hack to make inline testcases work.
    makefile = os.path.join(src_dir, "Makefile")
    if not os.path.isfile(makefile):
        makefile = os.path.join(build_dir, "Makefile")
    return [make,
            "VPATH="+src_dir,
            "-C", build_dir,
            "-I", src_dir,
            "-I", os.path.join(lldb_test, "make"),
            "-f", makefile]


def getArchSpec(architecture):
    """
    Helper function to return the key-value string to specify the architecture
    used for the make system.
    """
    arch = architecture if architecture else None
    if not arch and configuration.arch:
        arch = configuration.arch

    return ("ARCH=" + arch) if arch else ""


def getCCSpec(compiler):
    """
    Helper function to return the key-value string to specify the compiler
    used for the make system.
    """
    cc = compiler if compiler else None
    if not cc and configuration.compiler:
        cc = configuration.compiler
    if cc:
        return "CC=\"%s\"" % cc
    else:
        return ""

def getDsymutilSpec():
    """
    Helper function to return the key-value string to specify the dsymutil
    used for the make system.
    """
    if "DSYMUTIL" in os.environ:
        return "DSYMUTIL={}".format(os.environ["DSYMUTIL"])
    return "";

def getSDKRootSpec():
    """
    Helper function to return the key-value string to specify the SDK root
    used for the make system.
    """
    if "SDKROOT" in os.environ:
        return "SDKROOT={}".format(os.environ["SDKROOT"])
    return "";

def getModuleCacheSpec():
    """
    Helper function to return the key-value string to specify the clang
    module cache used for the make system.
    """
    if configuration.clang_module_cache_dir:
        return "CLANG_MODULE_CACHE_DIR={}".format(
            configuration.clang_module_cache_dir)
    return "";

def getCmdLine(d):
    """
    Helper function to return a properly formatted command line argument(s)
    string used for the make system.
    """

    # If d is None or an empty mapping, just return an empty string.
    if not d:
        return ""
    pattern = '%s="%s"' if "win32" in sys.platform else "%s='%s'"

    def setOrAppendVariable(k, v):
        append_vars = ["CFLAGS", "CFLAGS_EXTRAS", "LD_EXTRAS"]
        if k in append_vars and k in os.environ:
            v = os.environ[k] + " " + v
        return pattern % (k, v)
    cmdline = " ".join([setOrAppendVariable(k, v) for k, v in list(d.items())])

    return cmdline


def runBuildCommands(commands, sender):
    try:
        lldbtest.system(commands, sender=sender)
    except subprocess.CalledProcessError as called_process_error:
        # Convert to a build-specific error.
        # We don't do that in lldbtest.system() since that
        # is more general purpose.
        raise build_exception.BuildError(called_process_error)


def buildDefault(
        sender=None,
        architecture=None,
        compiler=None,
        dictionary=None,
        testdir=None,
        testname=None):
    """Build the binaries the default way."""
    commands = []
    commands.append(getMake(testdir, testname) +
                    ["all",
                     getArchSpec(architecture),
                     getCCSpec(compiler),
                     getDsymutilSpec(),
                     getSDKRootSpec(),
                     getModuleCacheSpec(),
                     getCmdLine(dictionary)])

    runBuildCommands(commands, sender=sender)

    # True signifies that we can handle building default.
    return True


def buildDwarf(
        sender=None,
        architecture=None,
        compiler=None,
        dictionary=None,
        testdir=None,
        testname=None):
    """Build the binaries with dwarf debug info."""
    commands = []
    commands.append(getMake(testdir, testname) +
                    ["MAKE_DSYM=NO",
                     getArchSpec(architecture),
                     getCCSpec(compiler),
                     getDsymutilSpec(),
                     getSDKRootSpec(),
                     getModuleCacheSpec(),
                     getCmdLine(dictionary)])

    runBuildCommands(commands, sender=sender)
    # True signifies that we can handle building dwarf.
    return True


def buildDwo(
        sender=None,
        architecture=None,
        compiler=None,
        dictionary=None,
        testdir=None,
        testname=None):
    """Build the binaries with dwarf debug info."""
    commands = []
    commands.append(getMake(testdir, testname) +
                    ["MAKE_DSYM=NO",
                     "MAKE_DWO=YES",
                     getArchSpec(architecture),
                     getCCSpec(compiler),
                     getDsymutilSpec(),
                     getSDKRootSpec(),
                     getModuleCacheSpec(),
                     getCmdLine(dictionary)])

    runBuildCommands(commands, sender=sender)
    # True signifies that we can handle building dwo.
    return True


def buildGModules(
        sender=None,
        architecture=None,
        compiler=None,
        dictionary=None,
        testdir=None,
        testname=None):
    """Build the binaries with dwarf debug info."""
    commands = []
    commands.append(getMake(testdir, testname) +
                    ["MAKE_DSYM=NO",
                     "MAKE_GMODULES=YES",
                     getArchSpec(architecture),
                     getCCSpec(compiler),
                     getDsymutilSpec(),
                     getSDKRootSpec(),
                     getModuleCacheSpec(),
                     getCmdLine(dictionary)])

    lldbtest.system(commands, sender=sender)
    # True signifies that we can handle building with gmodules.
    return True


def cleanup(sender=None, dictionary=None):
    """Perform a platform-specific cleanup after the test."""
    return True

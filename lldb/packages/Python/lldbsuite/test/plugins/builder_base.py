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

import os, sys
import platform
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil

def getArchitecture():
    """Returns the architecture in effect the test suite is running with."""
    return os.environ["ARCH"] if "ARCH" in os.environ else ""

def getCompiler():
    """Returns the compiler in effect the test suite is running with."""
    compiler = os.environ.get("CC", "clang")
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

def getMake():
    """Returns the name for GNU make"""
    if platform.system() == "FreeBSD" or platform.system() == "NetBSD":
      return "gmake"
    else:
      return "make"

def getArchSpec(architecture):
    """
    Helper function to return the key-value string to specify the architecture
    used for the make system.
    """
    arch = architecture if architecture else None
    if not arch and "ARCH" in os.environ:
        arch = os.environ["ARCH"]

    return ("ARCH=" + arch) if arch else ""

def getCCSpec(compiler):
    """
    Helper function to return the key-value string to specify the compiler
    used for the make system.
    """
    cc = compiler if compiler else None
    if not cc and "CC" in os.environ:
        cc = os.environ["CC"]
    if cc:
        return "CC=\"%s\"" % cc
    else:
        return ""

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
        append_vars = ["CFLAGS_EXTRAS", "LD_EXTRAS"]
        if k in append_vars and k in os.environ:
            v = os.environ[k] + " " + v
        return pattern % (k, v)
    cmdline = " ".join([setOrAppendVariable(k, v) for k, v in list(d.items())])

    return cmdline


def buildDefault(sender=None, architecture=None, compiler=None, dictionary=None, clean=True):
    """Build the binaries the default way."""
    commands = []
    if clean:
        commands.append([getMake(), "clean", getCmdLine(dictionary)])
    commands.append([getMake(), getArchSpec(architecture), getCCSpec(compiler), getCmdLine(dictionary)])

    lldbtest.system(commands, sender=sender)

    # True signifies that we can handle building default.
    return True

def buildDwarf(sender=None, architecture=None, compiler=None, dictionary=None, clean=True):
    """Build the binaries with dwarf debug info."""
    commands = []
    if clean:
        commands.append([getMake(), "clean", getCmdLine(dictionary)])
    commands.append([getMake(), "MAKE_DSYM=NO", getArchSpec(architecture), getCCSpec(compiler), getCmdLine(dictionary)])

    lldbtest.system(commands, sender=sender)
    # True signifies that we can handle building dwarf.
    return True

def buildDwo(sender=None, architecture=None, compiler=None, dictionary=None, clean=True):
    """Build the binaries with dwarf debug info."""
    commands = []
    if clean:
        commands.append([getMake(), "clean", getCmdLine(dictionary)])
    commands.append([getMake(), "MAKE_DSYM=NO", "MAKE_DWO=YES", getArchSpec(architecture), getCCSpec(compiler), getCmdLine(dictionary)])

    lldbtest.system(commands, sender=sender)
    # True signifies that we can handle building dwo.
    return True

def cleanup(sender=None, dictionary=None):
    """Perform a platform-specific cleanup after the test."""
    #import traceback
    #traceback.print_stack()
    commands = []
    if os.path.isfile("Makefile"):
        commands.append([getMake(), "clean", getCmdLine(dictionary)])

    lldbtest.system(commands, sender=sender)
    # True signifies that we can handle cleanup.
    return True

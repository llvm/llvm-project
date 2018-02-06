
from __future__ import print_function
import os
import lldbsuite.test.lldbtest as lldbtest

from builder_base import *

def buildDsym(
        sender=None,
        architecture=None,
        compiler=None,
        dictionary=None,
        clean=True,
        testdir=None,
        testname=None):
    """Build the binaries with dsym debug info."""
    commands = []
    if clean:
        commands.append(getMake(testdir, testname) +
                        ["clean", getCmdLine(dictionary)])
    commands.append(getMake(testdir, testname) +
                    ["MAKE_DSYM=YES",
                     getArchSpec(architecture),
                     getCCSpec(compiler),
                     "all", getCmdLine(dictionary)])

    runBuildCommands(commands, sender=sender)

    # True signifies that we can handle building dsym.
    return True

"""
Test the GDB remote reproducer.
"""

from __future__ import print_function

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemoteReproducer(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        """Test record and replay of gdb-remote packets."""
        self.build()

        # Create temp directory for the reproducer.
        exe = self.getBuildArtifact("a.out")

        # First capture a regular debugging session.
        self.runCmd("reproducer capture enable")

        reproducer_path = self.dbg.GetReproducerPath()

        self.runCmd("file {}".format(exe))
        self.runCmd("breakpoint set -f main.c -l 13")
        self.runCmd("run")
        self.runCmd("bt")
        self.runCmd("cont")

        # Generate the reproducer and stop capturing.
        self.runCmd("reproducer generate")
        self.runCmd("reproducer capture disable")

        # Replay the session from the reproducer.
        self.runCmd("reproducer replay {}".format(reproducer_path))

        # We have to issue the same commands.
        self.runCmd("file {}".format(exe))
        self.runCmd("breakpoint set -f main.c -l 13")
        self.runCmd("run")
        self.runCmd("bt")
        self.expect("cont")

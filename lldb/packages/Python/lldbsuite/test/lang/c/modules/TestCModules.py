"""Test that importing modules in C works as expected."""

from __future__ import print_function


from distutils.version import StrictVersion
import os
import time
import platform

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CModulesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfFreeBSD
    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="http://llvm.org/pr23456 'fopen' has unknown return type")
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24489: Name lookup not working correctly on Windows")
    @skipIf(macos_version=["<", "10.12"])
    def test_expr(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        self.expect(
            "expr -l objc++ -- @import Darwin; 3",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "int",
                "3"])

        self.expect(
            "expr *fopen(\"/dev/zero\", \"w\")",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "FILE",
                "_close"])

        self.expect("expr *myFile", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["a", "5", "b", "9"])

        self.expect(
            "expr MIN((uint64_t)2, (uint64_t)3)",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "uint64_t",
                "2"])

        self.expect("expr stdin", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["(FILE *)", "0x"])

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set breakpoint 0 here.')

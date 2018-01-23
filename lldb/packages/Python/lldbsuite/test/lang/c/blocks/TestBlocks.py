"""Test that lldb can invoke blocks and access variables inside them"""

from __future__ import print_function


import unittest2
import os
import time
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class BlocksTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    lines = []

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break at.
        self.lines.append(line_number('main.c', '// Set breakpoint 0 here.'))
        self.lines.append(line_number('main.c', '// Set breakpoint 1 here.'))

    def launch_common(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.is_started = False

        # Break inside the foo function which takes a bar_ptr argument.
        for line in self.lines:
            lldbutil.run_break_set_by_file_and_line(
                self, "main.c", line, num_expected_locations=1, loc_exact=True)

        self.wait_for_breakpoint()

    @skipUnlessDarwin
    def test_expr(self):
        self.launch_common()

        self.expect("expression a + b", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["= 7"])

        self.expect("expression c", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["= 1"])

        self.wait_for_breakpoint()

        # This should display correctly.
        self.expect("expression (int)neg (-12)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["= 12"])

    @skipUnlessDarwin
    def test_define(self):
        self.launch_common()

        self.runCmd(
            "expression int (^$add)(int, int) = ^int(int a, int b) { return a + b; };")
        self.expect(
            "expression $add(2,3)",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[" = 5"])

        self.runCmd("expression int $a = 3")
        self.expect(
            "expression int (^$addA)(int) = ^int(int b) { return $a + b; };",
            "Proper error is reported on capture",
            error=True)

    def wait_for_breakpoint(self):
        if not self.is_started:
            self.is_started = True
            self.runCmd("process launch", RUN_SUCCEEDED)
        else:
            self.runCmd("process continue", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

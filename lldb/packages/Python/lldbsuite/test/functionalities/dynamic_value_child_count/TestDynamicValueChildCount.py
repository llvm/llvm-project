"""
Test that dynamic values update their child count correctly
"""

from __future__ import print_function


import os
import time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DynamicValueChildCountTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        # Find the line number to break for main.c.

        self.main_third_call_line = line_number(
            'pass-to-base.cpp', '// Break here and check b has 0 children')
        self.main_fourth_call_line = line_number(
            'pass-to-base.cpp', '// Break here and check b still has 0 children')
        self.main_fifth_call_line = line_number(
            'pass-to-base.cpp', '// Break here and check b has one child now')
        self.main_sixth_call_line = line_number(
            'pass-to-base.cpp', '// Break here and check b has 0 children again')

    @add_test_categories(['pyapi'])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24663")
    def test_get_dynamic_vals(self):
        """Test fetching C++ dynamic values from pointers & references."""
        """Get argument vals for the call stack when stopped on a breakpoint."""
        self.build(dictionary=self.getBuildFlags())
        exe = self.getBuildArtifact("a.out")

        # Create a target from the debugger.

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set up our breakpoints:

        third_call_bpt = target.BreakpointCreateByLocation(
            'pass-to-base.cpp', self.main_third_call_line)
        self.assertTrue(third_call_bpt,
                        VALID_BREAKPOINT)
        fourth_call_bpt = target.BreakpointCreateByLocation(
            'pass-to-base.cpp', self.main_fourth_call_line)
        self.assertTrue(fourth_call_bpt,
                        VALID_BREAKPOINT)
        fifth_call_bpt = target.BreakpointCreateByLocation(
            'pass-to-base.cpp', self.main_fifth_call_line)
        self.assertTrue(fifth_call_bpt,
                        VALID_BREAKPOINT)
        sixth_call_bpt = target.BreakpointCreateByLocation(
            'pass-to-base.cpp', self.main_sixth_call_line)
        self.assertTrue(sixth_call_bpt,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        b = self.frame().FindVariable("b").GetDynamicValue(lldb.eDynamicCanRunTarget)
        self.assertTrue(b.GetNumChildren() == 0, "b has 0 children")
        self.runCmd("continue")
        self.assertTrue(b.GetNumChildren() == 0, "b still has 0 children")
        self.runCmd("continue")
        self.assertTrue(b.GetNumChildren() != 0, "b now has 1 child")
        self.runCmd("continue")
        self.assertTrue(
            b.GetNumChildren() == 0,
            "b didn't go back to 0 children")

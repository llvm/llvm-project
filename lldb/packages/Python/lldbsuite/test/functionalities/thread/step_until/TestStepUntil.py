"""Test stepping over vrs. hitting breakpoints & subsequent stepping in various forms."""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCStepping(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def getCategories(self):
        return ['basic_process']

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers that we will step to in main:
        self.main_source = "main.c"
        self.less_than_two = line_number('main.c', 'Less than 2')
        self.greater_than_two = line_number('main.c', 'Greater than or equal to 2.')
        self.back_out_in_main = line_number('main.c', 'Back out in main')

    def do_until (self, args, until_lines, expected_linenum):
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        main_source_spec = lldb.SBFileSpec(self.main_source)
        break_before = target.BreakpointCreateBySourceRegex(
            'At the start',
            main_source_spec)
        self.assertTrue(break_before, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            args, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, break_before)

        if len(threads) != 1:
            self.fail("Failed to stop at first breakpoint in main.")

        thread = threads[0]
        return thread

        thread = self.common_setup(None)

        cmd_interp = self.dbg.GetCommandInterpreter()
        ret_obj = lldb.SBCommandReturnObject()

        cmd_line = "thread until"
        for line_num in until_lines:
            cmd_line += " %d"%(line_num)
 
        cmd_interp.HandleCommand(cmd_line, ret_obj)
        self.assertTrue(ret_obj.Succeeded(), "'%s' failed: %s."%(cmd_line, ret_obj.GetError()))

        frame = thread.frames[0]
        line = frame.GetLineEntry().GetLine()
        self.assertEqual(line, expected_linenum, 'Did not get the expected stop line number')

    def test_hitting_one (self):
        """Test thread step until - targeting one line and hitting it."""
        self.do_until(None, [self.less_than_two], self.less_than_two)

    def test_targetting_two_hitting_first (self):
        """Test thread step until - targeting two lines and hitting one."""
        self.do_until(["foo", "bar", "baz"], [self.less_than_two, self.greater_than_two], self.greater_than_two)

    def test_targetting_two_hitting_second (self):
        """Test thread step until - targeting two lines and hitting the other one."""
        self.do_until(None, [self.less_than_two, self.greater_than_two], self.less_than_two)

    def test_missing_one (self):
        """Test thread step until - targeting one line and missing it."""
        self.do_until(["foo", "bar", "baz"], [self.less_than_two], self.back_out_in_main)




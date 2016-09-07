"""
Test thread step-in, step-over and step-out work with the "Avoid no debug" option.
"""

from __future__ import print_function


import os
import re
import sys

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ReturnValueTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    def test_step_out_with_python(self):
        """Test stepping out using avoid-no-debug with dsyms."""
        self.build()
        self.get_to_starting_point()
        self.do_step_out_past_nodebug()

    @add_test_categories(['pyapi'])
    @decorators.expectedFailureAll(
        compiler="gcc", bugnumber="llvm.org/pr28549")
    @decorators.expectedFailureAll(
        compiler="clang",
        compiler_version=[
            ">=",
            "3.9"],
        archs=["i386"],
        bugnumber="llvm.org/pr28549")
    def test_step_over_with_python(self):
        """Test stepping over using avoid-no-debug with dwarf."""
        self.build()
        self.get_to_starting_point()
        self.do_step_over_past_nodebug()

    @add_test_categories(['pyapi'])
    @decorators.expectedFailureAll(
        compiler="gcc", bugnumber="llvm.org/pr28549")
    @decorators.expectedFailureAll(
        compiler="clang",
        compiler_version=[
            ">=",
            "3.9"],
        archs=["i386"],
        bugnumber="llvm.org/pr28549")
    def test_step_in_with_python(self):
        """Test stepping in using avoid-no-debug with dwarf."""
        self.build()
        self.get_to_starting_point()
        self.do_step_in_past_nodebug()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "with-debug.c"
        self.main_source_spec = lldb.SBFileSpec("with-debug.c")
        self.dbg.HandleCommand(
            "settings set target.process.thread.step-out-avoid-nodebug true")

    def tearDown(self):
        self.dbg.HandleCommand(
            "settings set target.process.thread.step-out-avoid-nodebug false")
        TestBase.tearDown(self)

    def hit_correct_line(self, pattern):
        target_line = line_number(self.main_source, pattern)
        self.assertTrue(
            target_line != 0,
            "Could not find source pattern " +
            pattern)
        cur_line = self.thread.frames[0].GetLineEntry().GetLine()
        self.assertTrue(
            cur_line == target_line,
            "Stepped to line %d instead of expected %d with pattern '%s'." %
            (cur_line,
             target_line,
             pattern))

    def hit_correct_function(self, pattern):
        name = self.thread.frames[0].GetFunctionName()
        self.assertTrue(
            pattern in name, "Got to '%s' not the expected function '%s'." %
            (name, pattern))

    def get_to_starting_point(self):
        exe = os.path.join(os.getcwd(), "a.out")
        error = lldb.SBError()

        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)

        inner_bkpt = self.target.BreakpointCreateBySourceRegex(
            "Stop here and step out of me", self.main_source_spec)
        self.assertTrue(inner_bkpt, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        self.process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(self.process, PROCESS_IS_VALID)

        # Now finish, and make sure the return value is correct.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            self.process, inner_bkpt)
        self.assertTrue(len(threads) == 1, "Stopped at inner breakpoint.")
        self.thread = threads[0]

    def do_step_out_past_nodebug(self):
        # The first step out takes us to the called_from_nodebug frame, just to make sure setting
        # step-out-avoid-nodebug doesn't change the behavior in frames with
        # debug info.
        self.thread.StepOut()
        self.hit_correct_line(
            "intermediate_return_value = called_from_nodebug_actual(some_value)")
        self.thread.StepOut()
        self.hit_correct_line(
            "int return_value = no_debug_caller(5, called_from_nodebug)")

    def do_step_over_past_nodebug(self):
        self.thread.StepOver()
        self.hit_correct_line(
            "intermediate_return_value = called_from_nodebug_actual(some_value)")
        self.thread.StepOver()
        self.hit_correct_line("return intermediate_return_value")
        self.thread.StepOver()
        # Note, lldb doesn't follow gdb's distinction between "step-out" and "step-over/step-in"
        # when exiting a frame.  In all cases we leave the pc at the point where we exited the
        # frame.  In gdb, step-over/step-in move to the end of the line they stepped out to.
        # If we ever change this we will need to fix this test.
        self.hit_correct_line(
            "int return_value = no_debug_caller(5, called_from_nodebug)")

    def do_step_in_past_nodebug(self):
        self.thread.StepInto()
        self.hit_correct_line(
            "intermediate_return_value = called_from_nodebug_actual(some_value)")
        self.thread.StepInto()
        self.hit_correct_line("return intermediate_return_value")
        self.thread.StepInto()
        # Note, lldb doesn't follow gdb's distinction between "step-out" and "step-over/step-in"
        # when exiting a frame.  In all cases we leave the pc at the point where we exited the
        # frame.  In gdb, step-over/step-in move to the end of the line they stepped out to.
        # If we ever change this we will need to fix this test.
        self.hit_correct_line(
            "int return_value = no_debug_caller(5, called_from_nodebug)")

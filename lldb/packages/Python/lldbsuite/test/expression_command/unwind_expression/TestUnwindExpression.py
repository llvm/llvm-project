"""
Test stopping at a breakpoint in an expression, and unwinding from there.
"""

from __future__ import print_function


import unittest2

import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class UnwindFromExpressionTest(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    @add_test_categories(['pyapi'])
    @expectedFailureAll(oslist=["windows"])
    def test_unwind_expression(self):
        """Test unwinding from an expression."""
        self.build()

        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create the breakpoint.
        main_spec = lldb.SBFileSpec("main.cpp", False)
        breakpoint = target.BreakpointCreateBySourceRegex(
            "// Set a breakpoint here to get started", main_spec)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        if not process:
            self.fail("SBTarget.LaunchProcess() failed")

        if process.GetState() != lldb.eStateStopped:
            self.fail("Process should be in the 'stopped' state, "
                      "instead the actual state is: '%s'" %
                      lldbutil.state_type_to_str(process.GetState()))

        thread = lldbutil.get_one_thread_stopped_at_breakpoint(
            process, breakpoint)
        self.assertIsNotNone(
            thread, "Expected one thread to be stopped at the breakpoint")

        #
        # Use Python API to evaluate expressions while stopped in a stack frame.
        #
        main_frame = thread.GetFrameAtIndex(0)

        # Next set a breakpoint in this function, set up Expression options to stop on
        # breakpoint hits, and call the function.
        fun_bkpt = target.BreakpointCreateBySourceRegex(
            "// Stop inside the function here.", main_spec)
        self.assertTrue(fun_bkpt, VALID_BREAKPOINT)
        options = lldb.SBExpressionOptions()
        options.SetIgnoreBreakpoints(False)
        options.SetUnwindOnError(False)

        val = main_frame.EvaluateExpression("a_function_to_call()", options)

        self.assertTrue(
            val.GetError().Fail(),
            "We did not complete the execution.")
        error_str = val.GetError().GetCString()
        self.assertTrue(
            "Execution was interrupted, reason: breakpoint" in error_str,
            "And the reason was right.")

        thread = lldbutil.get_one_thread_stopped_at_breakpoint(
            process, fun_bkpt)
        self.assertTrue(
            thread.IsValid(),
            "We are indeed stopped at our breakpoint")

        # Now unwind the expression, and make sure we got back to where we
        # started.
        error = thread.UnwindInnermostExpression()
        self.assertTrue(error.Success(), "We succeeded in unwinding")

        cur_frame = thread.GetFrameAtIndex(0)
        self.assertTrue(
            cur_frame.IsEqual(main_frame),
            "We got back to the main frame.")

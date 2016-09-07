"""
Test that expr will time out and allow other threads to run if it blocks.
"""

from __future__ import print_function


import os
import time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprDoesntDeadlockTestCase(TestBase):

    def getCategories(self):
        return ['basic_process']

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(oslist=['freebsd'], bugnumber='llvm.org/pr17946')
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="Windows doesn't have pthreads, test needs to be ported")
    def test_with_run_command(self):
        """Test that expr will time out and allow other threads to run if it blocks."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint at source line before call_me_to_get_lock
        # gets called.

        main_file_spec = lldb.SBFileSpec("locking.c")
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Break here', main_file_spec)
        if self.TraceOn():
            print("breakpoint:", breakpoint)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be on self.line1 and the break condition should hold.
        from lldbsuite.test.lldbutil import get_stopped_thread
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition")

        frame0 = thread.GetFrameAtIndex(0)

        var = frame0.EvaluateExpression("call_me_to_get_lock()")
        self.assertTrue(var.IsValid())
        self.assertTrue(var.GetValueAsSigned(0) == 567)

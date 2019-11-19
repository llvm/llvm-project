"""
Test calling a function that hits a signal set to auto-restart, make sure the call completes.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprCommandThatRestartsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.main_source = "lotta-signals.c"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    @skipIfFreeBSD  # llvm.org/pr19246: intermittent failure
    @skipIfDarwin  # llvm.org/pr19246: intermittent failure
    @skipIfWindows  # Test relies on signals, unsupported on Windows
    @expectedFlakeyAndroid(bugnumber="llvm.org/pr19246")
    @expectedFailureNetBSD
    def test(self):
        """Test calling function that hits a signal and restarts."""
        self.build()
        self.call_function()

    def check_after_call(self, num_sigchld):
        after_call = self.sigchld_no.GetValueAsSigned(-1)
        self.assertTrue(
            after_call -
            self.start_sigchld_no == num_sigchld,
            "Really got %d SIGCHLD signals through the call." %
            (num_sigchld))
        self.start_sigchld_no = after_call

        # Check that we are back where we were before:
        frame = self.thread.GetFrameAtIndex(0)
        self.assertTrue(
            self.orig_frame_pc == frame.GetPC(),
            "Restored the zeroth frame correctly")

    def call_function(self):
        (target, process, self.thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                      'Stop here in main.', self.main_source_spec)

        # Make sure the SIGCHLD behavior is pass/no-stop/no-notify:
        return_obj = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            "process handle SIGCHLD -s 0 -p 1 -n 0", return_obj)
        self.assertTrue(return_obj.Succeeded(), "Set SIGCHLD to pass, no-stop")

        # The sigchld_no variable should be 0 at this point.
        self.sigchld_no = target.FindFirstGlobalVariable("sigchld_no")
        self.assertTrue(
            self.sigchld_no.IsValid(),
            "Got a value for sigchld_no")

        self.start_sigchld_no = self.sigchld_no.GetValueAsSigned(-1)
        self.assertTrue(
            self.start_sigchld_no != -1,
            "Got an actual value for sigchld_no")

        options = lldb.SBExpressionOptions()
        # processing 30 signals takes a while, increase the expression timeout
        # a bit
        options.SetTimeoutInMicroSeconds(3000000)  # 3s
        options.SetUnwindOnError(True)

        frame = self.thread.GetFrameAtIndex(0)
        # Store away the PC to check that the functions unwind to the right
        # place after calls
        self.orig_frame_pc = frame.GetPC()

        num_sigchld = 30
        value = frame.EvaluateExpression(
            "call_me (%d)" %
            (num_sigchld), options)
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
        self.assertTrue(value.GetValueAsSigned(-1) == num_sigchld)

        self.check_after_call(num_sigchld)

        # Okay, now try with a breakpoint in the called code in the case where
        # we are ignoring breakpoint hits.
        handler_bkpt = target.BreakpointCreateBySourceRegex(
            "Got sigchld %d.", self.main_source_spec)
        self.assertTrue(handler_bkpt.GetNumLocations() > 0)
        options.SetIgnoreBreakpoints(True)
        options.SetUnwindOnError(True)

        value = frame.EvaluateExpression(
            "call_me (%d)" %
            (num_sigchld), options)

        self.assertTrue(value.IsValid() and value.GetError().Success())
        self.assertTrue(value.GetValueAsSigned(-1) == num_sigchld)
        self.check_after_call(num_sigchld)

        # Now set the signal to print but not stop and make sure that calling
        # still works:
        self.dbg.GetCommandInterpreter().HandleCommand(
            "process handle SIGCHLD -s 0 -p 1 -n 1", return_obj)
        self.assertTrue(
            return_obj.Succeeded(),
            "Set SIGCHLD to pass, no-stop, notify")

        value = frame.EvaluateExpression(
            "call_me (%d)" %
            (num_sigchld), options)

        self.assertTrue(value.IsValid() and value.GetError().Success())
        self.assertTrue(value.GetValueAsSigned(-1) == num_sigchld)
        self.check_after_call(num_sigchld)

        # Now set this unwind on error to false, and make sure that we still
        # complete the call:
        options.SetUnwindOnError(False)
        value = frame.EvaluateExpression(
            "call_me (%d)" %
            (num_sigchld), options)

        self.assertTrue(value.IsValid() and value.GetError().Success())
        self.assertTrue(value.GetValueAsSigned(-1) == num_sigchld)
        self.check_after_call(num_sigchld)

        # Okay, now set UnwindOnError to true, and then make the signal behavior to stop
        # and see that now we do stop at the signal point:

        self.dbg.GetCommandInterpreter().HandleCommand(
            "process handle SIGCHLD -s 1 -p 1 -n 1", return_obj)
        self.assertTrue(
            return_obj.Succeeded(),
            "Set SIGCHLD to pass, stop, notify")

        value = frame.EvaluateExpression(
            "call_me (%d)" %
            (num_sigchld), options)
        self.assertTrue(
            value.IsValid() and value.GetError().Success() == False)

        # Set signal handling back to no-stop, and continue and we should end
        # up back in out starting frame:
        self.dbg.GetCommandInterpreter().HandleCommand(
            "process handle SIGCHLD -s 0 -p 1 -n 1", return_obj)
        self.assertTrue(
            return_obj.Succeeded(),
            "Set SIGCHLD to pass, no-stop, notify")

        error = process.Continue()
        self.assertTrue(
            error.Success(),
            "Continuing after stopping for signal succeeds.")

        frame = self.thread.GetFrameAtIndex(0)
        self.assertTrue(
            frame.GetPC() == self.orig_frame_pc,
            "Continuing returned to the place we started.")

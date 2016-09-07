"""
Test calling a function that throws an ObjC exception, make sure that it doesn't propagate the exception.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprCommandWithThrowTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.main_source = "call-throws.m"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    @skipUnlessDarwin
    def test(self):
        """Test calling a function that throws and ObjC exception."""
        self.build()
        self.call_function()

    def check_after_call(self):
        # Check that we are back where we were before:
        frame = self.thread.GetFrameAtIndex(0)
        self.assertTrue(
            self.orig_frame_pc == frame.GetPC(),
            "Restored the zeroth frame correctly")

    def call_function(self):
        """Test calling function that throws."""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateBySourceRegex(
            'I am about to throw.', self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]

        options = lldb.SBExpressionOptions()
        options.SetUnwindOnError(True)

        frame = self.thread.GetFrameAtIndex(0)
        # Store away the PC to check that the functions unwind to the right
        # place after calls
        self.orig_frame_pc = frame.GetPC()

        value = frame.EvaluateExpression("[my_class callMeIThrow]", options)
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success() == False)

        self.check_after_call()

        # Okay, now try with a breakpoint in the called code in the case where
        # we are ignoring breakpoint hits.
        handler_bkpt = target.BreakpointCreateBySourceRegex(
            "I felt like it", self.main_source_spec)
        self.assertTrue(handler_bkpt.GetNumLocations() > 0)
        options.SetIgnoreBreakpoints(True)
        options.SetUnwindOnError(True)

        value = frame.EvaluateExpression("[my_class callMeIThrow]", options)

        self.assertTrue(
            value.IsValid() and value.GetError().Success() == False)
        self.check_after_call()

        # Now set the ObjC language breakpoint and make sure that doesn't
        # interfere with the call:
        exception_bkpt = target.BreakpointCreateForException(
            lldb.eLanguageTypeObjC, False, True)
        self.assertTrue(exception_bkpt.GetNumLocations() > 0)

        options.SetIgnoreBreakpoints(True)
        options.SetUnwindOnError(True)

        value = frame.EvaluateExpression("[my_class callMeIThrow]", options)

        self.assertTrue(
            value.IsValid() and value.GetError().Success() == False)
        self.check_after_call()

        # Now turn off exception trapping, and call a function that catches the exceptions,
        # and make sure the function actually completes, and we get the right
        # value:
        options.SetTrapExceptions(False)
        value = frame.EvaluateExpression("[my_class iCatchMyself]", options)
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
        self.assertTrue(value.GetValueAsUnsigned() == 57)
        self.check_after_call()
        options.SetTrapExceptions(True)

        # Now set this unwind on error to false, and make sure that we stop
        # where the exception was thrown
        options.SetUnwindOnError(False)
        value = frame.EvaluateExpression("[my_class callMeIThrow]", options)

        self.assertTrue(
            value.IsValid() and value.GetError().Success() == False)
        self.check_after_call()

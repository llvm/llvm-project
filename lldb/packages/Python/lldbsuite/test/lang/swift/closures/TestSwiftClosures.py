"""
Test that we can print and call closures passed in various contexts
"""

from __future__ import print_function


import os
import time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestPassedClosures(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(bugnumber="rdar://31816998")
    def test_static_closure_type(self):
        """This tests that we can print a closure with statically known return type."""
        self.build()
        self.static_type(False)

    @expectedFailureAll(bugnumber="rdar://31816998")
    def test_static_closure_call(self):
        """This tests that we can call a closure with statically known return type."""
        self.build()
        self.static_type(True)

    @expectedFailureAll(bugnumber="rdar://31816998")
    def test_generic_closure_type(self):
        """This tests that we can print a closure with generic return type."""
        self.build()
        self.generic_type(False)

    @expectedFailureAll(bugnumber="rdar://31816998")
    def test_generic_closure_call(self):
        """This tests that we can call a closure with generic return type."""
        self.build()
        self.generic_type(True)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def get_to_bkpt (self, bkpt_name):
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint in main.c at the source matching
        # "Set a breakpoint here"
        breakpoint = target.BreakpointCreateBySourceRegex(
            bkpt_name, lldb.SBFileSpec("main.swift"))
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() >= 1,
                        VALID_BREAKPOINT)

        error = lldb.SBError()
        # This is the launch info.  If you want to launch with arguments or
        # environment variables, add them using SetArguments or
        # SetEnvironmentEntries

        launch_info = lldb.SBLaunchInfo(None)
        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        # Did we hit our breakpoint?
        from lldbsuite.test.lldbutil import get_threads_stopped_at_breakpoint
        threads = get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertTrue(
            len(threads) == 1,
            "There should be a thread stopped at our breakpoint")

        # The hit count for the breakpoint should be 1.
        self.assertTrue(breakpoint.GetHitCount() == 1)

        self.frame = threads[0].GetFrameAtIndex(0)

    def static_type(self, test_call):

        self.get_to_bkpt("break here for static type")
        opts = lldb.SBExpressionOptions()

        if not test_call:
            # First see that we can print the function we were passed:
            result = self.frame.EvaluateExpression("fn", opts)
            error = result.GetError()
            self.assertTrue(error.Success(),"'fn' failed: %s"%(error.GetCString()))
            self.assertTrue("() -> Swift.Int" in result.GetValue(), "Got the function name wrong: %s."%(result.GetValue()))
            self.assertTrue("() -> Swift.Int" in result.GetTypeName(), "Got the function type wrong: %s."%(result.GetTypeName()))
        
        if test_call:
            # Now see that we can call it:
            result = self.frame.EvaluateExpression("fn()", opts)
            error.result.GetError()
            self.assertTrue(error.Success(),"'fn()' failed: %s"%(error.GetCString()))
            self.assertTrue(result.GetValue() == "3", "Got the wrong value: %s"%(result.GetValue()))

    def generic_type(self, test_call):
        self.get_to_bkpt("break here for generic type")
        opts = lldb.SBExpressionOptions()

        if not test_call:
            # First see that we can print the function we were passed:
            result = frame.EvaluateExpression("fn", opts)
            error = result.GetError()
            self.assertTrue(error.Success(),"'fn' failed: %s"%(error.GetCString()))
            self.assertTrue("() -> A" in result.GetValue(), "Got the function name wrong: %s."%(result.GetValue()))
            self.assertTrue("() -> A" in result.GetTypeName(), "Got the function type wrong: %s."%(result.GetTypeName()))
        
        if test_call:
            # Now see that we can call it:
            result = frame.EvaluateExpression("fn()", opts)
            error.result.GetError()
            self.assertTrue(error.Success(),"'fn()' failed: %s"%(error.GetCString()))
            self.assertTrue(result.GetValue() == "3", "Got the wrong value: %s"%(result.GetValue()))




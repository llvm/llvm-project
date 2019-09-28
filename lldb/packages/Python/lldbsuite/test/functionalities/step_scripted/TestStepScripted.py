"""
Tests stepping with scripted thread plans.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *

class StepScriptedTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def test_standard_step_out(self):
        """Tests stepping with the scripted thread plan laying over a standard thread plan for stepping out."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.step_out_with_scripted_plan("Steps.StepOut")

    def test_scripted_step_out(self):
        """Tests stepping with the scripted thread plan laying over an another scripted thread plan for stepping out."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.step_out_with_scripted_plan("Steps.StepScripted")

    def setUp(self):
        TestBase.setUp(self)
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.runCmd("command script import Steps.py")

    def step_out_with_scripted_plan(self, name):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                                                            "Set a breakpoint here",
                                                                            self.main_source_file)

        frame = thread.GetFrameAtIndex(0)
        self.assertEqual("foo", frame.GetFunctionName())

        err = thread.StepUsingScriptedThreadPlan(name)
        self.assertTrue(err.Success(), err.GetCString())

        frame = thread.GetFrameAtIndex(0)
        self.assertEqual("main", frame.GetFunctionName())

    def test_misspelled_plan_name(self):
        """Test that we get a useful error if we misspell the plan class name"""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                                                            "Set a breakpoint here",
                                                                            self.main_source_file)
        stop_id = process.GetStopID()
        # Pass a non-existent class for the plan class:
        err = thread.StepUsingScriptedThreadPlan("NoSuchModule.NoSuchPlan")
        
        # Make sure we got a good error:
        self.assertTrue(err.Fail(), "We got a failure state")
        msg = err.GetCString()
        self.assertTrue("NoSuchModule.NoSuchPlan" in msg, "Mentioned missing class")
        
        # Make sure we didn't let the process run:
        self.assertEqual(stop_id, process.GetStopID(), "Process didn't run")

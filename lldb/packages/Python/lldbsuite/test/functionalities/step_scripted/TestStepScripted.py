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
        self.runCmd("command script import Steps.py")

    def step_out_with_scripted_plan(self, name):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self, "Set a breakpoint here", self.main_source_file)

        frame = thread.GetFrameAtIndex(0)
        self.assertEqual("foo", frame.GetFunctionName())

        err = thread.StepUsingScriptedThreadPlan(name)
        self.assertTrue(err.Success(), err.GetCString())

        frame = thread.GetFrameAtIndex(0)
        self.assertEqual("main", frame.GetFunctionName())

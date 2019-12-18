"""
Test that stop hooks trigger on "step-out"
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestStopHooks(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # If your test case doesn't stress debug info, the
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_stop_hooks_step_out(self):
        """Test that stop hooks fire on step-out."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.step_out_test()

    def step_out_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file)

        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand("target stop-hook add -o 'expr g_var++'", result)
        self.assertTrue(result.Succeeded, "Set the target stop hook")
        thread.StepOut()
        var = target.FindFirstGlobalVariable("g_var")
        self.assertTrue(var.IsValid())
        self.assertEqual(var.GetValueAsUnsigned(), 1, "Updated g_var")



"""
Make sure the !N and !-N commands work properly.
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestHistoryRecall(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # If your test case doesn't stress debug info, the
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_history_recall(self):
        """Test the !N and !-N functionality of the command interpreter."""
        self.sample_test()

    def sample_test(self):
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand("command history", result, True)
        interp.HandleCommand("platform list", result, True)

        interp.HandleCommand("!0", result, False)
        self.assertTrue(result.Succeeded(), "!0 command did not work: %s"%(result.GetError()))
        self.assertTrue("command history" in result.GetOutput(), "!0 didn't rerun command history")

        interp.HandleCommand("!-1", result, False)
        self.assertTrue(result.Succeeded(), "!-1 command did not work: %s"%(result.GetError()))
        self.assertTrue("host:" in result.GetOutput(), "!-1 didn't rerun platform list.")

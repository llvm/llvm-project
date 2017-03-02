"""
Describe the purpose of the test class here.
"""

from __future__ import print_function


import os
import time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class RenameThisSampleTestTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # If your test case doesn't stress debug info, the 
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_sample_rename_this(self):
        """There can be many tests in a test case - describe this test here."""
        self.build()
        self.sample_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def sample_test(self):
        """You might use the test implementation in several ways, say so here."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint in main.c at the source matching
        # "Set a breakpoint here"
        breakpoint = target.BreakpointCreateBySourceRegex(
            "Set a breakpoint here", lldb.SBFileSpec("main.c"))
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

        frame = threads[0].GetFrameAtIndex(0)
        test_var = frame.FindVariable("test_var")
        self.assertTrue(test_var.GetError().Success(), "Failed to fetch test_var")
        test_value = test_var.GetValueAsUnsigned()
        self.assertEqual(test_value, 10, "Got the right value for test_var")


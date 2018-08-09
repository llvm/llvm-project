"""
Make sure the frame variable -g, -a, and -l flags work.
"""

from __future__ import print_function


import os
import time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestFrameVar(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # If your test case doesn't stress debug info, the
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_var(self):
        self.build()
        self.do_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def do_test(self):
        exe = self.getBuildArtifact("a.out")

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
        command_result = lldb.SBCommandReturnObject()
        interp = self.dbg.GetCommandInterpreter()

        # Just get args:
        result = interp.HandleCommand("frame var -l", command_result)
        self.assertEqual(result, lldb.eReturnStatusSuccessFinishResult, "frame var -a didn't succeed")
        output = command_result.GetOutput()
        self.assertTrue("argc" in output, "Args didn't find argc")
        self.assertTrue("argv" in output, "Args didn't find argv")
        self.assertTrue("test_var" not in output, "Args found a local")
        self.assertTrue("g_var" not in output, "Args found a global")

        # Just get locals:
        result = interp.HandleCommand("frame var -a", command_result)
        self.assertEqual(result, lldb.eReturnStatusSuccessFinishResult, "frame var -a didn't succeed")
        output = command_result.GetOutput()
        self.assertTrue("argc" not in output, "Locals found argc")
        self.assertTrue("argv" not in output, "Locals found argv")
        self.assertTrue("test_var" in output, "Locals didn't find test_var")
        self.assertTrue("g_var" not in output, "Locals found a global")

        # Get the file statics:
        result = interp.HandleCommand("frame var -l -a -g", command_result)
        self.assertEqual(result, lldb.eReturnStatusSuccessFinishResult, "frame var -a didn't succeed")
        output = command_result.GetOutput()
        self.assertTrue("argc" not in output, "Globals found argc")
        self.assertTrue("argv" not in output, "Globals found argv")
        self.assertTrue("test_var" not in output, "Globals found test_var")
        self.assertTrue("g_var" in output, "Globals didn't find g_var")




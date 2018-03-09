"""
Test to ensure SBFrame::Disassemble produces SOME output
"""

from __future__ import print_function


import os
import time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class FrameDisassembleTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_disassemble(self):
        """Sample test to ensure SBFrame::Disassemble produces SOME output."""
        self.build()
        self.frame_disassemble_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def frame_disassemble_test(self):
        """Sample test to ensure SBFrame::Disassemble produces SOME output"""
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint in main.c at the source matching
        # "Set a breakpoint here"
        breakpoint = target.BreakpointCreateBySourceRegex(
            "Set a breakpoint here", lldb.SBFileSpec("main.cpp"))
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
        disassembly = frame.Disassemble()
        self.assertTrue(len(disassembly) != 0, "Disassembly was empty.")

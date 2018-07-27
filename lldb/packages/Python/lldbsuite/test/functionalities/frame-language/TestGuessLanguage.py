"""
Test the SB API SBFrame::GuessLanguage.
"""

from __future__ import print_function


import os
import time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestFrameGuessLanguage(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # If your test case doesn't stress debug info, the
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr37658")
    def test_guess_language(self):
        """Test GuessLanguage for C and C++."""
        self.build()
        self.do_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def check_language(self, thread, frame_no, test_lang):
        frame = thread.frames[frame_no]
        self.assertTrue(frame.IsValid(), "Frame %d was not valid."%(frame_no))
        lang = frame.GuessLanguage()
        self.assertEqual(lang, test_lang)

    def do_test(self):
        """Test GuessLanguage for C & C++."""
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint in main.c at the source matching
        # "Set a breakpoint here"
        breakpoint = target.BreakpointCreateBySourceRegex(
            "Set breakpoint here", lldb.SBFileSpec("somefunc.c"))
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

        thread = threads[0]

        c_frame_language = lldb.eLanguageTypeC99
        # gcc emits DW_LANG_C89 even if -std=c99 was specified
        if "gcc" in self.getCompiler():
            c_frame_language = lldb.eLanguageTypeC89

        self.check_language(thread, 0, c_frame_language)
        self.check_language(thread, 1, lldb.eLanguageTypeC_plus_plus)
        self.check_language(thread, 2, lldb.eLanguageTypeC_plus_plus)




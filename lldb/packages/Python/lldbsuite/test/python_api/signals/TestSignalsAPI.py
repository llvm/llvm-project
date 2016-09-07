"""
Test SBProcess APIs, including ReadMemory(), WriteMemory(), and others.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbutil import get_stopped_thread, state_type_to_str


class SignalsAPITestCase(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    @skipIfWindows  # Windows doesn't have signals
    def test_ignore_signal(self):
        """Test Python SBUnixSignals.Suppress/Stop/Notify() API."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        line = line_number(
            "main.cpp",
            "// Set break point at this line and setup signal ignores.")
        breakpoint = target.BreakpointCreateByLocation("main.cpp", line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint")

        unix_signals = process.GetUnixSignals()
        sigint = unix_signals.GetSignalNumberFromName("SIGINT")
        unix_signals.SetShouldSuppress(sigint, True)
        unix_signals.SetShouldStop(sigint, False)
        unix_signals.SetShouldNotify(sigint, False)

        process.Continue()
        self.assertTrue(
            process.state == lldb.eStateExited,
            "The process should have exited")
        self.assertTrue(
            process.GetExitStatus() == 0,
            "The process should have returned 0")

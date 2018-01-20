"""Test the Go Data Formatter Plugin."""

import os
import time
import unittest2
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGoLanguage(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfFreeBSD  # llvm.org/pr24895 triggers assertion failure
    @skipIfRemote  # Not remote test suite ready
    @no_debug_info_test
    @skipUnlessGoInstalled
    def test_go_formatter_plugin(self):
        """Test go data formatters."""
        self.buildGo()
        self.launchProcess()
        self.check_formatters()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "main.go"
        self.break_line = line_number(self.main_source, '// stop here')

    def launchProcess(self):
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        self.bpt = target.BreakpointCreateByLocation(
            self.main_source, self.break_line)
        self.assertTrue(self.bpt, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        thread_list = lldbutil.get_threads_stopped_at_breakpoint(
            process, self.bpt)

        # Make sure we stopped at the first breakpoint.
        self.assertTrue(
            len(thread_list) != 0,
            "No thread stopped at our breakpoint.")
        self.assertTrue(len(thread_list) == 1,
                        "More than one thread stopped at our breakpoint.")

        frame = thread_list[0].GetFrameAtIndex(0)
        self.assertTrue(frame, "Got a valid frame 0 frame.")

    def check_formatters(self):
        a = self.frame().FindVariable('a')
        self.assertEqual('(string) a = "my string"', str(a))
        b = self.frame().FindVariable('b')
        self.assertEqual(
            "([]int) b = (len 2, cap 7) {\n  [0] = 0\n  [1] = 0\n}",
            str(b))


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

"""Test the Go OS Plugin."""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGoASTContext(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    @skipIfFreeBSD  # llvm.org/pr24895 triggers assertion failure
    @skipIfRemote  # Not remote test suite ready
    @no_debug_info_test
    @skipUnlessGoInstalled
    def test_goroutine_plugin(self):
        """Test goroutine as threads support."""
        self.buildGo()
        self.launchProcess()
        self.check_goroutines()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "main.go"
        self.break_line1 = line_number(self.main_source, '// stop1')
        self.break_line2 = line_number(self.main_source, '// stop2')
        self.break_line3 = line_number(self.main_source, '// stop3')

    def launchProcess(self):
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        self.bpt1 = target.BreakpointCreateByLocation(
            self.main_source, self.break_line1)
        self.assertTrue(self.bpt1, VALID_BREAKPOINT)
        self.bpt2 = target.BreakpointCreateByLocation(
            self.main_source, self.break_line2)
        self.assertTrue(self.bpt2, VALID_BREAKPOINT)
        self.bpt3 = target.BreakpointCreateByLocation(
            self.main_source, self.break_line3)
        self.assertTrue(self.bpt3, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        thread_list = lldbutil.get_threads_stopped_at_breakpoint(
            process, self.bpt1)

        # Make sure we stopped at the first breakpoint.
        self.assertTrue(
            len(thread_list) != 0,
            "No thread stopped at our breakpoint.")
        self.assertTrue(len(thread_list) == 1,
                        "More than one thread stopped at our breakpoint.")

        frame = thread_list[0].GetFrameAtIndex(0)
        self.assertTrue(frame, "Got a valid frame 0 frame.")

    def check_goroutines(self):
        self.assertLess(len(self.process().threads), 20)
        self.process().Continue()

        # Make sure we stopped at the 2nd breakpoint
        thread_list = lldbutil.get_threads_stopped_at_breakpoint(
            self.process(), self.bpt2)
        self.assertTrue(
            len(thread_list) != 0,
            "No thread stopped at our breakpoint.")
        self.assertTrue(len(thread_list) == 1,
                        "More than one thread stopped at our breakpoint.")

        # There's (at least) 21 goroutines.
        self.assertGreater(len(self.process().threads), 20)
        # self.dbg.HandleCommand("log enable lldb os")

        # Now test that stepping works if the memory thread moves to a
        # different backing thread.
        for i in list(range(11)):
            self.thread().StepOver()
            self.assertEqual(
                lldb.eStopReasonPlanComplete,
                self.thread().GetStopReason(),
                self.thread().GetStopDescription(100))

        # Disable the plugin and make sure the goroutines disappear
        self.dbg.HandleCommand(
            "settings set plugin.os.goroutines.enable false")
        self.thread().StepInstruction(False)
        self.assertLess(len(self.process().threads), 20)

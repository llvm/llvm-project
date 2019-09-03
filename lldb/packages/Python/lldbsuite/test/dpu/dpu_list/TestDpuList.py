"""
Test dpu_list command
"""

from __future__ import print_function

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbutil import get_stopped_thread

import dpu_commands


class DpuListTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    def test_dpu_list(self):
        """Use Python APIs to check dpu_list command."""
        self.build()
        self.do_test_dpu_list()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def do_test_dpu_list(self):
        exe = self.getBuildArtifact("host.out")
        dpu_exe = self.getBuildArtifact("a.out")
        filespec = lldb.SBFileSpec("host.c", False)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint_launch = \
            target.BreakpointCreateByName("dpu_launch_thread_on_dpu")

        env = ["%s=%s" % (k, v) for k, v in os.environ.iteritems()]
        process = target.LaunchSimple(
            None, env, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        thread = process.GetSelectedThread()
        self.assertTrue(thread.GetStopReason() == lldb.eStopReasonBreakpoint)
        target.BreakpointDelete(breakpoint_launch.GetID())
        while thread.GetFrameAtIndex(0).GetFunctionName() \
                != "dpu_launch_thread_on_dpu":
            thread.StepOut()
        thread.StepOut()

        breakpoint_poll = target.BreakpointCreateByName("dpu_poll_dpu")
        process.Continue()
        self.assertTrue(thread.GetStopReason() == lldb.eStopReasonBreakpoint)
        target.BreakpointDelete(breakpoint_poll.GetID())

        process.GetSelectedThread().SetSelectedFrame(1)
        dpu_list = dpu_commands.dpu_list(self.dbg, "", None, None)

        for (dpu_addr, dpu_id, dpu_status, dpu_program) in dpu_list:
            if dpu_status == "RUNNING":
                self.assertTrue(dpu_program == "'" + dpu_exe + "'")
                return
        assert False

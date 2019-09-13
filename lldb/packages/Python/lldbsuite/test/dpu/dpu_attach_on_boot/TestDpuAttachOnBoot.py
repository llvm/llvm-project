"""
Test dpu_attach_on_boot command
"""

from __future__ import print_function

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbutil import get_stopped_thread

import dpu_commands


class DpuAttachOnBootTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    def test_dpu_attach_on_boot(self):
        """Use Python APIs to check dpu_attach_on_boot command."""
        self.build()
        self.do_test_dpu_attach_on_boot()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def do_test_dpu_attach_on_boot(self):
        exe = self.getBuildArtifact("host.out")
        filespec = lldb.SBFileSpec("host.c", False)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint_main = \
            target.BreakpointCreateByName("main")

        env = ["%s=%s" % (k, v) for k, v in os.environ.iteritems()]
        process = target.LaunchSimple(
            None, env, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        thread = process.GetSelectedThread()
        self.assertTrue(thread.GetStopReason() == lldb.eStopReasonBreakpoint)
        target.BreakpointDelete(breakpoint_main.GetID())

        target_dpu = dpu_commands.dpu_attach_on_boot(self.dbg, "", None, None)
        self.assertTrue(target_dpu.IsValid())
        dpu_commands.dpu_detach(self.dbg, None, None, None)

        target_dpu = dpu_commands.dpu_attach_on_boot(self.dbg, "", None, None)
        self.assertTrue(target_dpu.IsValid())
        dpu_commands.dpu_detach(self.dbg, None, None, None)

        dpu_list = dpu_commands.dpu_list(self.dbg, "", None, None)

        print(dpu_list[0][0])
        target_dpu = dpu_commands.dpu_attach_on_boot(self.dbg,
                                                     str(dpu_list[0][0]),
                                                     None, None)
        self.assertTrue(target_dpu.IsValid())
        dpu_commands.dpu_detach(self.dbg, None, None, None)

        target_dpu = dpu_commands.dpu_attach_on_boot(self.dbg,
                                                     str(dpu_list[0][0]),
                                                     None, None)
        self.assertTrue(target_dpu.IsValid())
        dpu_commands.dpu_detach(self.dbg, None, None, None)

        process.Continue()
        self.assertTrue(process.GetState() == lldb.eStateExited)
        self.assertTrue(process.GetExitStatus() == 0x0)

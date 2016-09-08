"""
Test inferior restart when breakpoint is set on running target.
"""

import os
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BreakpointSetRestart(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    BREAKPOINT_TEXT = 'Set a breakpoint here'

    def test_breakpoint_set_restart(self):
        self.build()

        cwd = self.get_process_working_directory()
        exe = os.path.join(cwd, "a.out")
        target = self.dbg.CreateTarget(exe)

        self.dbg.SetAsync(True)
        process = target.LaunchSimple(None, None, cwd)

        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), [
                lldb.eStateRunning])
        bp = target.BreakpointCreateBySourceRegex(
            self.BREAKPOINT_TEXT, lldb.SBFileSpec(
                os.path.join(
                    cwd, 'main.cpp')))
        self.assertTrue(
            bp.IsValid() and bp.GetNumLocations() == 1,
            VALID_BREAKPOINT)

        event = lldb.SBEvent()
        while self.dbg.GetListener().WaitForEvent(2, event):
            if lldb.SBProcess.GetStateFromEvent(
                    event) == lldb.eStateStopped and lldb.SBProcess.GetRestartedFromEvent(event):
                continue
            if lldb.SBProcess.GetStateFromEvent(event) == lldb.eStateRunning:
                continue
            self.fail(
                "Setting a breakpoint generated an unexpected event: %s" %
                lldb.SBDebugger.StateAsCString(
                    lldb.SBProcess.GetStateFromEvent(event)))

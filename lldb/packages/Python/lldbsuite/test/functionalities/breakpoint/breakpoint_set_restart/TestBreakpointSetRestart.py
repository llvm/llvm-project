"""
Test inferior restart when breakpoint is set on running target.
"""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class BreakpointSetRestart(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    BREAKPOINT_TEXT = 'Set a breakpoint here'

    def test_breakpoint_set_restart(self):
        self.build()

        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        self.dbg.SetAsync(True)
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        event = lldb.SBEvent()
        # Wait for inferior to transition to running state
        while self.dbg.GetListener().WaitForEvent(2, event):
            if lldb.SBProcess.GetStateFromEvent(event) == lldb.eStateRunning:
                break

        bp = target.BreakpointCreateBySourceRegex(
            self.BREAKPOINT_TEXT, lldb.SBFileSpec('main.cpp'))
        self.assertTrue(
            bp.IsValid() and bp.GetNumLocations() == 1,
            VALID_BREAKPOINT)

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

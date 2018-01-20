"""
Test that the breakpoint auto-continue flag works correctly.
"""

from __future__ import print_function


import os
import time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class BreakpointAutoContinue(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def test_breakpoint_auto_continue(self):
        """Make sure the auto continue continues with no other complications"""
        self.build()
        self.simple_auto_continue()

    def test_auto_continue_with_command(self):
        """Add a command, make sure the command gets run"""
        self.build()
        self.auto_continue_with_command()

    def test_auto_continue_on_location(self):
        """Set auto-continue on a location and make sure only that location continues"""
        self.build()
        self.auto_continue_location()

    def make_target_and_bkpt(self, additional_options=None, num_expected_loc=1, 
                             pattern="Set a breakpoint here"):
        exe = self.getBuildArtifact("a.out")
        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target.IsValid(), "Target is not valid")
        
        extra_options_txt = "--auto-continue 1 "
        if additional_options:
            extra_options_txt += additional_options
        bpno = lldbutil.run_break_set_by_source_regexp(self, pattern, 
                                            extra_options = extra_options_txt, 
                                            num_expected_locations = num_expected_loc)
        return bpno

    def launch_it (self, expected_state):
        error = lldb.SBError()
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())

        process = self.target.Launch(launch_info, error)
        self.assertTrue(error.Success(), "Launch failed.")

        state = process.GetState()
        self.assertEqual(state, expected_state, "Didn't get expected state")

        return process

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def simple_auto_continue(self):
        bpno = self.make_target_and_bkpt()
        process = self.launch_it(lldb.eStateExited)

        bkpt = self.target.FindBreakpointByID(bpno)
        self.assertEqual(bkpt.GetHitCount(), 2, "Should have run through the breakpoint twice")

    def auto_continue_with_command(self):
        bpno = self.make_target_and_bkpt("-N BKPT -C 'break modify --auto-continue 0 BKPT'")
        process = self.launch_it(lldb.eStateStopped)
        state = process.GetState()
        self.assertEqual(state, lldb.eStateStopped, "Process should be stopped")
        bkpt = self.target.FindBreakpointByID(bpno)
        threads = lldbutil.get_threads_stopped_at_breakpoint(process, bkpt)
        self.assertEqual(len(threads), 1, "There was a thread stopped at our breakpoint")
        self.assertEqual(bkpt.GetHitCount(), 2, "Should have hit the breakpoint twice")

    def auto_continue_location(self):
        bpno = self.make_target_and_bkpt(pattern="Set a[^ ]* breakpoint here", num_expected_loc=2)
        bkpt = self.target.FindBreakpointByID(bpno)
        bkpt.SetAutoContinue(False)

        loc = lldb.SBBreakpointLocation()
        for i in range(0,2):
            func_name = bkpt.location[i].GetAddress().function.name
            if func_name == "main":
                loc = bkpt.location[i]

        self.assertTrue(loc.IsValid(), "Didn't find a location in main")
        loc.SetAutoContinue(True)

        process = self.launch_it(lldb.eStateStopped)

        threads = lldbutil.get_threads_stopped_at_breakpoint(process, bkpt)
        self.assertEqual(len(threads), 1, "Didn't get one thread stopped at our breakpoint")
        func_name = threads[0].frame[0].function.name
        self.assertEqual(func_name, "call_me")

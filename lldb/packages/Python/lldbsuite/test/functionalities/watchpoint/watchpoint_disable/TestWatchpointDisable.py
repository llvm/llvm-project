"""
Test that the SBWatchpoint::SetEnable API works.
"""

import os
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbplatform, lldbplatformutil


class TestWatchpointSetEnable(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    # Watchpoints not supported
    @expectedFailureAndroid(archs=['arm', 'aarch64'])
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24446: WINDOWS XFAIL TRIAGE - Watchpoints not supported on Windows")
    def test_disable_works (self):
        """Set a watchpoint, disable it, and make sure it doesn't get hit."""
        self.build()
        self.do_test(False)

    @expectedFailureAndroid(archs=['arm', 'aarch64'])
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24446: WINDOWS XFAIL TRIAGE - Watchpoints not supported on Windows")
    def test_disable_enable_works (self):
        """Set a watchpoint, disable it, and make sure it doesn't get hit."""
        self.build()
        self.do_test(True)

    def do_test(self, test_enable):
        """Set a watchpoint, disable it and make sure it doesn't get hit."""

        exe = 'a.out'

        exe = os.path.join(os.getcwd(), exe)
        main_file_spec = lldb.SBFileSpec("main.c")

        # Create a target by the debugger.
        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)
        cwd = os.getcwd()
        

        bkpt_before = self.target.BreakpointCreateBySourceRegex("Set a breakpoint here", main_file_spec)
        self.assertEqual(bkpt_before.GetNumLocations(),  1, "Failed setting the before breakpoint.")

        bkpt_after = self.target.BreakpointCreateBySourceRegex("We should have stopped", main_file_spec)
        self.assertEqual(bkpt_after.GetNumLocations(), 1, "Failed setting the after breakpoint.")

        process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        thread = lldbutil.get_one_thread_stopped_at_breakpoint(process, bkpt_before)
        self.assertTrue(thread.IsValid(), "We didn't stop at the before breakpoint.")

        ret_val = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand("watchpoint set variable -w write global_var", ret_val)
        self.assertTrue(ret_val.Succeeded(), "Watchpoint set variable did not return success.")

        wp = self.target.FindWatchpointByID(1)
        self.assertTrue(wp.IsValid(), "Didn't make a valid watchpoint.")
        self.assertTrue(wp.GetWatchAddress() != lldb.LLDB_INVALID_ADDRESS, "Watch address is invalid")

        wp.SetEnabled(False)
        self.assertTrue(not wp.IsEnabled(), "The watchpoint thinks it is still enabled")
        
        process.Continue()
        
        stop_reason = thread.GetStopReason()

        self.assertEqual(stop_reason, lldb.eStopReasonBreakpoint, "We didn't stop at our breakpoint.")

        if test_enable:
            wp.SetEnabled(True)
            self.assertTrue(wp.IsEnabled(), "The watchpoint thinks it is still disabled.")
            process.Continue()
            stop_reason = thread.GetStopReason()
            self.assertEqual(stop_reason, lldb.eStopReasonWatchpoint, "We didn't stop at our watchpoint")
        

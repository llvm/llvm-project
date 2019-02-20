"""
Test thread states.
"""

from __future__ import print_function


import unittest2
import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ThreadStateTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="llvm.org/pr15824 thread states not properly maintained")
    @skipIfDarwin # llvm.org/pr15824 thread states not properly maintained and <rdar://problem/28557237>
    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr18190 thread states not properly maintained")
    def test_state_after_breakpoint(self):
        """Test thread state after breakpoint."""
        self.build(dictionary=self.getBuildFlags(use_cpp11=False))
        self.thread_state_after_breakpoint_test()

    @skipIfDarwin  # 'llvm.org/pr23669', cause Python crash randomly
    @expectedFailureAll(
        oslist=lldbplatformutil.getDarwinOSTriples(),
        bugnumber="llvm.org/pr23669")
    @expectedFailureAll(oslist=["freebsd"], bugnumber="llvm.org/pr15824")
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24660")
    def test_state_after_continue(self):
        """Test thread state after continue."""
        self.build(dictionary=self.getBuildFlags(use_cpp11=False))
        self.thread_state_after_continue_test()

    @skipIfDarwin  # 'llvm.org/pr23669', cause Python crash randomly
    @expectedFailureDarwin('llvm.org/pr23669')
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24660")
    # thread states not properly maintained
    @unittest2.expectedFailure("llvm.org/pr16712")
    def test_state_after_expression(self):
        """Test thread state after expression."""
        self.build(dictionary=self.getBuildFlags(use_cpp11=False))
        self.thread_state_after_expression_test()

    # thread states not properly maintained
    @unittest2.expectedFailure("llvm.org/pr15824 and <rdar://problem/28557237>")
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24668: Breakpoints not resolved correctly")
    @skipIfDarwin # llvm.org/pr15824 thread states not properly maintained and <rdar://problem/28557237>
    def test_process_state(self):
        """Test thread states (comprehensive)."""
        self.build(dictionary=self.getBuildFlags(use_cpp11=False))
        self.thread_states_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers for our breakpoints.
        self.break_1 = line_number('main.cpp', '// Set first breakpoint here')
        self.break_2 = line_number('main.cpp', '// Set second breakpoint here')

    def thread_state_after_breakpoint_test(self):
        """Test thread state after breakpoint."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint in the main thread.
        bp = lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.break_1, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Get the target process
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        # Make sure the thread is in the stopped state.
        self.assertTrue(
            thread.IsStopped(),
            "Thread state isn't \'stopped\' during breakpoint 1.")
        self.assertFalse(thread.IsSuspended(),
                         "Thread state is \'suspended\' during breakpoint 1.")

        # Kill the process
        self.runCmd("process kill")

    def wait_for_running_event(self, process):
        listener = self.dbg.GetListener()
        if lldb.remote_platform:
            lldbutil.expect_state_changes(
                self, listener, process, [
                    lldb.eStateConnected])
        lldbutil.expect_state_changes(
            self, listener, process, [
                lldb.eStateRunning])

    def thread_state_after_continue_test(self):
        """Test thread state after continue."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint in the main thread.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.break_1, num_expected_locations=1)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.break_2, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Get the target process
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        # Continue, the inferior will go into an infinite loop waiting for
        # 'g_test' to change.
        self.dbg.SetAsync(True)
        self.runCmd("continue")
        self.wait_for_running_event(process)

        # Check the thread state. It should be running.
        self.assertFalse(
            thread.IsStopped(),
            "Thread state is \'stopped\' when it should be running.")
        self.assertFalse(
            thread.IsSuspended(),
            "Thread state is \'suspended\' when it should be running.")

        # Go back to synchronous interactions
        self.dbg.SetAsync(False)

        # Kill the process
        self.runCmd("process kill")

    def thread_state_after_expression_test(self):
        """Test thread state after expression."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint in the main thread.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.break_1, num_expected_locations=1)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.break_2, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Get the target process
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        # Get the inferior out of its loop
        self.runCmd("expression g_test = 1")

        # Check the thread state
        self.assertTrue(
            thread.IsStopped(),
            "Thread state isn't \'stopped\' after expression evaluation.")
        self.assertFalse(
            thread.IsSuspended(),
            "Thread state is \'suspended\' after expression evaluation.")

        # Let the process run to completion
        self.runCmd("process continue")

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24668: Breakpoints not resolved correctly")
    @skipIfDarwin # llvm.org/pr15824 thread states not properly maintained and <rdar://problem/28557237>
    @no_debug_info_test
    def test_process_interrupt(self):
        """Test process interrupt and continue."""
        self.build(dictionary=self.getBuildFlags(use_cpp11=False))
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint in the main thread.
        bpno = lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.break_1, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Get the target process
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        # Remove the breakpoint to avoid the single-step-over-bkpt dance in the
        # "continue" below
        self.assertTrue(target.BreakpointDelete(bpno))

        # Continue, the inferior will go into an infinite loop waiting for
        # 'g_test' to change.
        self.dbg.SetAsync(True)
        self.runCmd("continue")
        self.wait_for_running_event(process)

        # Go back to synchronous interactions
        self.dbg.SetAsync(False)

        # Stop the process
        self.runCmd("process interrupt")

        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonSignal)

        # Get the inferior out of its loop
        self.runCmd("expression g_test = 1")

        # Run to completion
        self.runCmd("continue")

    def thread_states_test(self):
        """Test thread states (comprehensive)."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint in the main thread.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.break_1, num_expected_locations=1)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.break_2, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Get the target process
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        # Make sure the thread is in the stopped state.
        self.assertTrue(
            thread.IsStopped(),
            "Thread state isn't \'stopped\' during breakpoint 1.")
        self.assertFalse(thread.IsSuspended(),
                         "Thread state is \'suspended\' during breakpoint 1.")

        # Continue, the inferior will go into an infinite loop waiting for
        # 'g_test' to change.
        self.dbg.SetAsync(True)
        self.runCmd("continue")
        self.wait_for_running_event(process)

        # Check the thread state. It should be running.
        self.assertFalse(
            thread.IsStopped(),
            "Thread state is \'stopped\' when it should be running.")
        self.assertFalse(
            thread.IsSuspended(),
            "Thread state is \'suspended\' when it should be running.")

        # Go back to synchronous interactions
        self.dbg.SetAsync(False)

        # Stop the process
        self.runCmd("process interrupt")

        self.assertEqual(thread.GetState(), lldb.eStopReasonSignal)

        # Check the thread state
        self.assertTrue(
            thread.IsStopped(),
            "Thread state isn't \'stopped\' after process stop.")
        self.assertFalse(thread.IsSuspended(),
                         "Thread state is \'suspended\' after process stop.")

        # Get the inferior out of its loop
        self.runCmd("expression g_test = 1")

        # Check the thread state
        self.assertTrue(
            thread.IsStopped(),
            "Thread state isn't \'stopped\' after expression evaluation.")
        self.assertFalse(
            thread.IsSuspended(),
            "Thread state is \'suspended\' after expression evaluation.")

        self.assertEqual(thread.GetState(), lldb.eStopReasonSignal)

        # Run to breakpoint 2
        self.runCmd("continue")

        self.assertEqual(thread.GetState(), lldb.eStopReasonBreakpoint)

        # Make sure both threads are stopped
        self.assertTrue(
            thread.IsStopped(),
            "Thread state isn't \'stopped\' during breakpoint 2.")
        self.assertFalse(thread.IsSuspended(),
                         "Thread state is \'suspended\' during breakpoint 2.")

        # Run to completion
        self.runCmd("continue")

        # At this point, the inferior process should have exited.
        self.assertEqual(process.GetState(), lldb.eStateExited, PROCESS_EXITED)

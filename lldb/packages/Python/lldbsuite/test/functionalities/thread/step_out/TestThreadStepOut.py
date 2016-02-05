"""
Test stepping out from a function in a multi-threaded program.
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ThreadStepOutTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfLinux                              # Test occasionally times out on the Linux build bot
    @expectedFailureLinux("llvm.org/pr23477") # Test occasionally times out on the Linux build bot
    @expectedFailureFreeBSD("llvm.org/pr18066") # inferior does not exit
    @expectedFailureWindows # Test crashes
    def test_step_single_thread(self):
        """Test thread step out on one thread via command interpreter. """
        self.build(dictionary=self.getBuildFlags())
        self.step_out_test(self.step_out_single_thread_with_cmd)

    @skipIfLinux                              # Test occasionally times out on the Linux build bot
    @expectedFailureLinux("llvm.org/pr23477") # Test occasionally times out on the Linux build bot
    @expectedFailureFreeBSD("llvm.org/pr19347") # 2nd thread stops at breakpoint
    @expectedFailureWindows # Test crashes
    def test_step_all_threads(self):
        """Test thread step out on all threads via command interpreter. """
        self.build(dictionary=self.getBuildFlags())
        self.step_out_test(self.step_out_all_threads_with_cmd)

    @skipIfLinux                              # Test occasionally times out on the Linux build bot
    @expectedFailureLinux("llvm.org/pr23477") # Test occasionally times out on the Linux build bot
    @expectedFailureFreeBSD("llvm.org/pr19347")
    @expectedFailureWindows("llvm.org/pr24681")
    def test_python(self):
        """Test thread step out on one thread via Python API (dwarf)."""
        self.build(dictionary=self.getBuildFlags())
        self.step_out_test(self.step_out_with_python)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number for our breakpoint.
        self.breakpoint = line_number('main.cpp', '// Set breakpoint here')
        if "gcc" in self.getCompiler() or self.isIntelCompiler(): 
            self.step_out_destination = line_number('main.cpp', '// Expect to stop here after step-out (icc and gcc)')
        else:
            self.step_out_destination = line_number('main.cpp', '// Expect to stop here after step-out (clang)')

    def step_out_single_thread_with_cmd(self):
        self.step_out_with_cmd("this-thread")
        self.expect("thread backtrace all", "Thread location after step out is correct",
            substrs = ["main.cpp:%d" % self.step_out_destination,
                       "main.cpp:%d" % self.breakpoint])

    def step_out_all_threads_with_cmd(self):
        self.step_out_with_cmd("all-threads")
        self.expect("thread backtrace all", "Thread location after step out is correct",
            substrs = ["main.cpp:%d" % self.step_out_destination])

    def step_out_with_cmd(self, run_mode):
        self.runCmd("thread select %d" % self.step_out_thread.GetIndexID())
        self.runCmd("thread step-out -m %s" % run_mode)
        self.expect("process status", "Expected stop reason to be step-out",
            substrs = ["stop reason = step out"])

        self.expect("thread list", "Selected thread did not change during step-out",
            substrs = ["* thread #%d" % self.step_out_thread.GetIndexID()])

    def step_out_with_python(self):
        self.step_out_thread.StepOut()

        reason = self.step_out_thread.GetStopReason()
        self.assertEqual(lldb.eStopReasonPlanComplete, reason,
            "Expected thread stop reason 'plancomplete', but got '%s'" % lldbutil.stop_reason_to_str(reason))

        # Verify location after stepping out
        frame = self.step_out_thread.GetFrameAtIndex(0)
        desc = lldbutil.get_description(frame.GetLineEntry())
        expect = "main.cpp:%d" % self.step_out_destination
        self.assertTrue(expect in desc, "Expected %s but thread stopped at %s" % (expect, desc))

    def step_out_test(self, step_out_func):
        """Test single thread step out of a function."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint in the main thread.
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.breakpoint, num_expected_locations=1)

        # The breakpoint list should show 1 location.
        self.expect("breakpoint list -f", "Breakpoint location shown correctly",
            substrs = ["1: file = 'main.cpp', line = %d, exact_match = 0, locations = 1" % self.breakpoint])

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Get the target process
        self.inferior_target = self.dbg.GetSelectedTarget()
        self.inferior_process = self.inferior_target.GetProcess()

        # Get the number of threads, ensure we see all three.
        num_threads = self.inferior_process.GetNumThreads()
        self.assertEqual(num_threads, 3, 'Number of expected threads and actual threads do not match.')

        (breakpoint_threads, other_threads) = ([], [])
        lldbutil.sort_stopped_threads(self.inferior_process,
                                      breakpoint_threads=breakpoint_threads,
                                      other_threads=other_threads)

        while len(breakpoint_threads) < 2:
            self.runCmd("thread continue %s" % " ".join([str(x.GetIndexID()) for x in other_threads]))
            lldbutil.sort_stopped_threads(self.inferior_process,
                                          breakpoint_threads=breakpoint_threads,
                                          other_threads=other_threads)

        self.step_out_thread = breakpoint_threads[0]

        # Step out of thread stopped at breakpoint
        step_out_func()

        # Run to completion
        self.runCmd("continue")

        # At this point, the inferior process should have exited.
        self.assertTrue(self.inferior_process.GetState() == lldb.eStateExited, PROCESS_EXITED)

"""
Test basics of mini dump debugging.
"""

from __future__ import print_function



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil

@skipUnlessWindows  # for now mini-dump debugging is limited to Windows hosts
class MiniDumpTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_process_info_in_mini_dump(self):
        """Test that lldb can read the process information from the minidump."""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertEqual(self.process.GetNumThreads(), 1)
        self.assertEqual(self.process.GetProcessID(), 4440)

    @no_debug_info_test
    def test_thread_info_in_mini_dump(self):
        """Test that lldb can read the thread information from the minidump."""
        # This process crashed due to an access violation (0xc0000005) in its one and only thread.
        self.assertEqual(self.process.GetNumThreads(), 1)
        thread = self.process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonException)
        stop_description = thread.GetStopDescription(256);
        self.assertTrue("0xc0000005" in stop_description);

    @no_debug_info_test
    def test_stack_info_in_mini_dump(self):
        """Test that we can see the stack."""
        self.assertEqual(self.process.GetNumThreads(), 1)
        thread = self.process.GetThreadAtIndex(0)
        # The crash is in main, so there should be one frame on the stack.
        self.assertEqual(thread.GetNumFrames(), 1)
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid())
        pc = frame.GetPC()
        eip = frame.FindRegister("pc")
        self.assertTrue(eip.IsValid())
        self.assertEqual(pc, eip.GetValueAsUnsigned())

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # target create -c fizzbuzz_no_heap.dmp
        self.dbg.CreateTarget("")
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore("fizzbuzz_no_heap.dmp")

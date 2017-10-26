"""
Test basics of mini dump debugging.
"""

from __future__ import print_function
from six import iteritems


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MiniDumpTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def test_process_info_in_mini_dump(self):
        """Test that lldb can read the process information from the minidump."""
        # target create -c fizzbuzz_no_heap.dmp
        self.dbg.CreateTarget("")
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore("fizzbuzz_no_heap.dmp")
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertEqual(self.process.GetNumThreads(), 1)
        self.assertEqual(self.process.GetProcessID(), 4440)

    def test_thread_info_in_mini_dump(self):
        """Test that lldb can read the thread information from the minidump."""
        # target create -c fizzbuzz_no_heap.dmp
        self.dbg.CreateTarget("")
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore("fizzbuzz_no_heap.dmp")
        # This process crashed due to an access violation (0xc0000005) in its
        # one and only thread.
        self.assertEqual(self.process.GetNumThreads(), 1)
        thread = self.process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonException)
        stop_description = thread.GetStopDescription(256)
        self.assertTrue("0xc0000005" in stop_description)

    def test_stack_info_in_mini_dump(self):
        """Test that we can see a trivial stack in a VS-generate mini dump."""
        # target create -c fizzbuzz_no_heap.dmp
        self.dbg.CreateTarget("")
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore("fizzbuzz_no_heap.dmp")
        self.assertEqual(self.process.GetNumThreads(), 1)
        thread = self.process.GetThreadAtIndex(0)

        pc_list = [ 0x00164d14, 0x00167c79, 0x00167e6d, 0x7510336a, 0x77759882, 0x77759855]

        self.assertEqual(thread.GetNumFrames(), len(pc_list))
        for i in range(len(pc_list)):
            frame = thread.GetFrameAtIndex(i)
            self.assertTrue(frame.IsValid())
            self.assertEqual(frame.GetPC(), pc_list[i])

    @skipUnlessWindows # Minidump saving works only on windows
    def test_deeper_stack_in_mini_dump(self):
        """Test that we can examine a more interesting stack in a mini dump."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        core = os.path.join(os.getcwd(), "core.dmp")
        try:
            # Set a breakpoint and capture a mini dump.
            target = self.dbg.CreateTarget(exe)
            breakpoint = target.BreakpointCreateByName("bar")
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory())
            self.assertEqual(process.GetState(), lldb.eStateStopped)
            self.assertTrue(process.SaveCore(core))
            self.assertTrue(os.path.isfile(core))
            self.assertTrue(process.Kill().Success())

            # Launch with the mini dump, and inspect the stack.
            target = self.dbg.CreateTarget(None)
            process = target.LoadCore(core)
            thread = process.GetThreadAtIndex(0)

            expected_stack = {0: 'bar', 1: 'foo', 2: 'main'}
            self.assertGreaterEqual(thread.GetNumFrames(), len(expected_stack))
            for index, name in iteritems(expected_stack):
                frame = thread.GetFrameAtIndex(index)
                self.assertTrue(frame.IsValid())
                function_name = frame.GetFunctionName()
                self.assertTrue(name in function_name)

        finally:
            # Clean up the mini dump file.
            self.assertTrue(self.dbg.DeleteTarget(target))
            if (os.path.isfile(core)):
                os.unlink(core)

    @skipUnlessWindows # Minidump saving works only on windows
    def test_local_variables_in_mini_dump(self):
        """Test that we can examine local variables in a mini dump."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        core = os.path.join(os.getcwd(), "core.dmp")
        try:
            # Set a breakpoint and capture a mini dump.
            target = self.dbg.CreateTarget(exe)
            breakpoint = target.BreakpointCreateByName("bar")
            process = target.LaunchSimple(
                None, None, self.get_process_working_directory())
            self.assertEqual(process.GetState(), lldb.eStateStopped)
            self.assertTrue(process.SaveCore(core))
            self.assertTrue(os.path.isfile(core))
            self.assertTrue(process.Kill().Success())

            # Launch with the mini dump, and inspect a local variable.
            target = self.dbg.CreateTarget(None)
            process = target.LoadCore(core)
            thread = process.GetThreadAtIndex(0)
            frame = thread.GetFrameAtIndex(0)
            value = frame.EvaluateExpression('x')
            self.assertEqual(value.GetValueAsSigned(), 3)

        finally:
            # Clean up the mini dump file.
            self.assertTrue(self.dbg.DeleteTarget(target))
            if (os.path.isfile(core)):
                os.unlink(core)

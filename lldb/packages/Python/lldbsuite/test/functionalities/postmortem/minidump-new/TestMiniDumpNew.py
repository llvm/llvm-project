"""
Test basics of Minidump debugging.
"""

from __future__ import print_function
from six import iteritems


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MiniDumpNewTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def test_process_info_in_minidump(self):
        """Test that lldb can read the process information from the Minidump."""
        # target create -c linux-x86_64.dmp
        self.dbg.CreateTarget(None)
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore("linux-x86_64.dmp")
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertEqual(self.process.GetNumThreads(), 1)
        self.assertEqual(self.process.GetProcessID(), 29917)

    def test_thread_info_in_minidump(self):
        """Test that lldb can read the thread information from the Minidump."""
        # target create -c linux-x86_64.dmp
        self.dbg.CreateTarget(None)
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore("linux-x86_64.dmp")
        # This process crashed due to a segmentation fault in its
        # one and only thread.
        self.assertEqual(self.process.GetNumThreads(), 1)
        thread = self.process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonSignal)
        stop_description = thread.GetStopDescription(256)
        self.assertTrue("SIGSEGV" in stop_description)

    def test_stack_info_in_minidump(self):
        """Test that we can see a trivial stack in a breakpad-generated Minidump."""
        # target create linux-x86_64 -c linux-x86_64.dmp
        self.dbg.CreateTarget("linux-x86_64")
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore("linux-x86_64.dmp")
        self.assertEqual(self.process.GetNumThreads(), 1)
        thread = self.process.GetThreadAtIndex(0)
        # frame #0: linux-x86_64`crash()
        # frame #1: linux-x86_64`_start
        self.assertEqual(thread.GetNumFrames(), 2)
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid())
        pc = frame.GetPC()
        eip = frame.FindRegister("pc")
        self.assertTrue(eip.IsValid())
        self.assertEqual(pc, eip.GetValueAsUnsigned())

    def test_snapshot_minidump(self):
        """Test that if we load a snapshot minidump file (meaning the process did not crash) there is no stop reason."""
        # target create -c linux-x86_64_not_crashed.dmp
        self.dbg.CreateTarget(None)
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore("linux-x86_64_not_crashed.dmp")
        self.assertEqual(self.process.GetNumThreads(), 1)
        thread = self.process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonNone)
        stop_description = thread.GetStopDescription(256)
        self.assertEqual(stop_description, None)

    def test_deeper_stack_in_minidump(self):
        """Test that we can examine a more interesting stack in a Minidump."""
        # Launch with the Minidump, and inspect the stack.
        # target create linux-x86_64_not_crashed -c linux-x86_64_not_crashed.dmp
        target = self.dbg.CreateTarget("linux-x86_64_not_crashed")
        process = target.LoadCore("linux-x86_64_not_crashed.dmp")
        thread = process.GetThreadAtIndex(0)

        expected_stack = {1: 'bar', 2: 'foo', 3: '_start'}
        self.assertGreaterEqual(thread.GetNumFrames(), len(expected_stack))
        for index, name in iteritems(expected_stack):
            frame = thread.GetFrameAtIndex(index)
            self.assertTrue(frame.IsValid())
            function_name = frame.GetFunctionName()
            self.assertTrue(name in function_name)

    def test_local_variables_in_minidump(self):
        """Test that we can examine local variables in a Minidump."""
        # Launch with the Minidump, and inspect a local variable.
        # target create linux-x86_64_not_crashed -c linux-x86_64_not_crashed.dmp
        target = self.dbg.CreateTarget("linux-x86_64_not_crashed")
        process = target.LoadCore("linux-x86_64_not_crashed.dmp")
        thread = process.GetThreadAtIndex(0)
        frame = thread.GetFrameAtIndex(1)
        value = frame.EvaluateExpression('x')
        self.assertEqual(value.GetValueAsSigned(), 3)

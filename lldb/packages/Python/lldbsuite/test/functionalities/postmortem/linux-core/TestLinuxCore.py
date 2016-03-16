"""
Test basics of linux core file debugging.
"""

from __future__ import print_function

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class LinuxCoreTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIf(bugnumber="llvm.org/pr26947")
    @no_debug_info_test
    def test_i386(self):
        """Test that lldb can read the process information from an i386 linux core file."""
        self.do_test("i386", 32306)

    @no_debug_info_test
    def test_x86_64(self):
        """Test that lldb can read the process information from an x86_64 linux core file."""
        self.do_test("x86_64", 32259)

    def do_test(self, arch, pid):
        target = self.dbg.CreateTarget(arch + ".out")
        process = target.LoadCore(arch + ".core")
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetNumThreads(), 1)
        self.assertEqual(process.GetProcessID(), pid)

        thread = process.GetSelectedThread()
        self.assertTrue(thread)
        self.assertEqual(thread.GetThreadID(), pid)
        backtrace = ["bar", "foo", "_start"]
        self.assertEqual(thread.GetNumFrames(), len(backtrace))
        for i in range(len(backtrace)):
            frame = thread.GetFrameAtIndex(i)
            self.assertTrue(frame)
            self.assertEqual(frame.GetFunctionName(), backtrace[i])
            self.assertEqual(frame.GetLineEntry().GetLine(),
                    line_number("main.c", "Frame " + backtrace[i]))
            self.assertEqual(frame.FindVariable("F").GetValueAsUnsigned(), ord(backtrace[i][0]))

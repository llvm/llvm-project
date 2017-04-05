"""
Test signal reporting when debugging with linux core files.
"""

from __future__ import print_function

import shutil
import struct

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LinuxCoreThreadsTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)
    _initial_platform = lldb.DBG.GetSelectedPlatform()

    _i386_pid = 5193
    _x86_64_pid = 5222

    # Thread id for the failing thread.
    _i386_tid = 5195
    _x86_64_tid = 5250

    @skipIf(oslist=['windows'])
    @skipIfDarwin # <rdar://problem/31380097>, fails started happening with r299199
    @skipIf(triple='^mips')
    def test_i386(self):
        """Test that lldb can read the process information from an i386 linux core file."""
        self.do_test("linux-i386", self._i386_pid, self._i386_tid)

    @skipIf(oslist=['windows'])
    @skipIfDarwin # <rdar://problem/31380097>, fails started happening with r299199
    @skipIf(triple='^mips')
    def test_x86_64(self):
        """Test that lldb can read the process information from an x86_64 linux core file."""
        self.do_test("linux-x86_64", self._x86_64_pid, self._x86_64_tid)

    def do_test(self, filename, pid, tid):
        target = self.dbg.CreateTarget("")
        process = target.LoadCore(filename + ".core")
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetNumThreads(), 3)
        self.assertEqual(process.GetProcessID(), pid)

        for thread in process:
            reason = thread.GetStopReason()
            if( thread.GetThreadID() == tid ):
                self.assertEqual(reason, lldb.eStopReasonSignal)
                signal = thread.GetStopReasonDataAtIndex(1)
                # Check we got signal 4 (SIGILL)
                self.assertEqual(signal, 4)
            else:
                signal = thread.GetStopReasonDataAtIndex(1)
                # Check we got no signal on the other threads
                self.assertEqual(signal, 0)

        self.dbg.DeleteTarget(target)
        lldb.DBG.SetSelectedPlatform(self._initial_platform)

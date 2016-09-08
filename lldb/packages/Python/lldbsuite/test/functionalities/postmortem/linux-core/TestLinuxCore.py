"""
Test basics of linux core file debugging.
"""

from __future__ import print_function

import shutil
import struct

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LinuxCoreTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)

    _i386_pid = 32306
    _x86_64_pid = 32259
    _s390x_pid = 1045

    @skipIf(bugnumber="llvm.org/pr26947")
    def test_i386(self):
        """Test that lldb can read the process information from an i386 linux core file."""
        self.do_test("i386", self._i386_pid)

    def test_x86_64(self):
        """Test that lldb can read the process information from an x86_64 linux core file."""
        self.do_test("x86_64", self._x86_64_pid)

    # This seems to hang on non-s390x platforms for some reason.  Disabling
    # for now.
    @skipIf(archs=no_match(['s390x']))
    def test_s390x(self):
        """Test that lldb can read the process information from an s390x linux core file."""
        self.do_test("s390x", self._s390x_pid)

    def test_same_pid_running(self):
        """Test that we read the information from the core correctly even if we have a running
        process with the same PID around"""
        try:
            shutil.copyfile("x86_64.out", "x86_64-pid.out")
            shutil.copyfile("x86_64.core", "x86_64-pid.core")
            with open("x86_64-pid.core", "r+b") as f:
                # These are offsets into the NT_PRSTATUS and NT_PRPSINFO structures in the note
                # segment of the core file. If you update the file, these offsets may need updating
                # as well. (Notes can be viewed with readelf --notes.)
                for pid_offset in [0x1c4, 0x320]:
                    f.seek(pid_offset)
                    self.assertEqual(
                        struct.unpack(
                            "<I",
                            f.read(4))[0],
                        self._x86_64_pid)

                    # We insert our own pid, and make sure the test still
                    # works.
                    f.seek(pid_offset)
                    f.write(struct.pack("<I", os.getpid()))
            self.do_test("x86_64-pid", os.getpid())
        finally:
            self.RemoveTempFile("x86_64-pid.out")
            self.RemoveTempFile("x86_64-pid.core")

    def test_two_cores_same_pid(self):
        """Test that we handle the situation if we have two core files with the same PID
        around"""
        alttarget = self.dbg.CreateTarget("altmain.out")
        altprocess = alttarget.LoadCore("altmain.core")
        self.assertTrue(altprocess, PROCESS_IS_VALID)
        self.assertEqual(altprocess.GetNumThreads(), 1)
        self.assertEqual(altprocess.GetProcessID(), self._x86_64_pid)

        altframe = altprocess.GetSelectedThread().GetFrameAtIndex(0)
        self.assertEqual(altframe.GetFunctionName(), "_start")
        self.assertEqual(
            altframe.GetLineEntry().GetLine(),
            line_number(
                "altmain.c",
                "Frame _start"))

        error = lldb.SBError()
        F = altprocess.ReadCStringFromMemory(
            altframe.FindVariable("F").GetValueAsUnsigned(), 256, error)
        self.assertTrue(error.Success())
        self.assertEqual(F, "_start")

        # without destroying this process, run the test which opens another core file with the
        # same pid
        self.do_test("x86_64", self._x86_64_pid)

    def do_test(self, filename, pid):
        target = self.dbg.CreateTarget(filename + ".out")
        process = target.LoadCore(filename + ".core")
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
            self.assertEqual(
                frame.FindVariable("F").GetValueAsUnsigned(), ord(
                    backtrace[i][0]))

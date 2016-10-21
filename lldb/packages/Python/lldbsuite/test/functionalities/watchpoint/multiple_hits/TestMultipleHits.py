"""
Test handling of cases when a single instruction triggers multiple watchpoints
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MultipleHitsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24446: WINDOWS XFAIL TRIAGE - Watchpoints not supported on Windows")
    @skipIf(bugnumber="llvm.org/pr30758", oslist=["linux"], archs=["arm", "aarch64"])
    def test(self):
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target and target.IsValid(), VALID_TARGET)

        bp = target.BreakpointCreateByName("main")
        self.assertTrue(bp and bp.IsValid(), "Breakpoint is valid")

        process = target.LaunchSimple(None, None,
                self.get_process_working_directory())
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame and frame.IsValid(), "Frame is valid")

        buf = frame.FindValue("buf", lldb.eValueTypeVariableGlobal)
        self.assertTrue(buf and buf.IsValid(), "buf is valid")

        for i in [0, target.GetAddressByteSize()]:
            member = buf.GetChildAtIndex(i)
            self.assertTrue(member and member.IsValid(), "member is valid")

            error = lldb.SBError()
            watch = member.Watch(True, True, True, error)
            self.assertTrue(error.Success())

        process.Continue();
        self.assertEqual(process.GetState(), lldb.eStateStopped)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonWatchpoint)


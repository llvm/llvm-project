"""
Tests ThreadSanitizer's support to detect a leaked thread.
"""

import os
import time
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import json


class TsanThreadLeakTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="non-core functionality, need to reenable and fix later (DES 2014.11.07)")
    @expectedFailureNetBSD
    @skipIfFreeBSD  # llvm.org/pr21136 runtimes not yet available by default
    @skipIfRemote
    @skipUnlessThreadSanitizer
    def test(self):
        self.build()
        self.tsan_tests()

    def tsan_tests(self):
        exe = self.getBuildArtifact("a.out")
        self.expect(
            "file " + exe,
            patterns=["Current executable set to .*a.out"])

        self.runCmd("run")

        stop_reason = self.dbg.GetSelectedTarget().process.GetSelectedThread().GetStopReason()
        if stop_reason == lldb.eStopReasonExec:
            # On OS X 10.10 and older, we need to re-exec to enable
            # interceptors.
            self.runCmd("continue")

        # the stop reason of the thread should be breakpoint.
        self.expect("thread list", "A thread leak should be detected",
                    substrs=['stopped', 'stop reason = Thread leak detected'])

        self.assertEqual(
            self.dbg.GetSelectedTarget().process.GetSelectedThread().GetStopReason(),
            lldb.eStopReasonInstrumentation)

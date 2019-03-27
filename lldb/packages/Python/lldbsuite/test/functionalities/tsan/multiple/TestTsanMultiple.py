"""
Test ThreadSanitizer when multiple different issues are found.
"""

import os
import time
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import json


class TsanMultipleTestCase(TestBase):

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

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def tsan_tests(self):
        exe = self.getBuildArtifact("a.out")
        self.expect(
            "file " + exe,
            patterns=["Current executable set to .*a.out"])

        self.runCmd("env TSAN_OPTIONS=abort_on_error=0")

        self.runCmd("run")

        stop_reason = self.dbg.GetSelectedTarget().process.GetSelectedThread().GetStopReason()
        if stop_reason == lldb.eStopReasonExec:
            # On OS X 10.10 and older, we need to re-exec to enable
            # interceptors.
            self.runCmd("continue")

        report_count = 0
        while self.dbg.GetSelectedTarget().process.GetSelectedThread(
        ).GetStopReason() == lldb.eStopReasonInstrumentation:
            report_count += 1

            stop_description = self.dbg.GetSelectedTarget(
            ).process.GetSelectedThread().GetStopDescription(100)

            self.assertTrue(
                (stop_description == "Data race detected") or
                (stop_description == "Use of deallocated memory detected") or
                (stop_description == "Thread leak detected") or
                (stop_description == "Use of an uninitialized or destroyed mutex detected") or
                (stop_description == "Unlock of an unlocked mutex (or by a wrong thread) detected")
            )

            self.expect(
                "thread info -s",
                "The extended stop info should contain the TSan provided fields",
                substrs=[
                    "instrumentation_class",
                    "description",
                    "mops"])

            output_lines = self.res.GetOutput().split('\n')
            json_line = '\n'.join(output_lines[2:])
            data = json.loads(json_line)
            self.assertEqual(data["instrumentation_class"], "ThreadSanitizer")

            backtraces = self.dbg.GetSelectedTarget().process.GetSelectedThread(
            ).GetStopReasonExtendedBacktraces(lldb.eInstrumentationRuntimeTypeThreadSanitizer)
            self.assertTrue(backtraces.GetSize() >= 1)

            self.runCmd("continue")

        self.assertEqual(
            self.dbg.GetSelectedTarget().process.GetState(),
            lldb.eStateExited,
            PROCESS_EXITED)

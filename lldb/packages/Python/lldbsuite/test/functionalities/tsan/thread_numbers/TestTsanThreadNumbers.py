"""
Tests that TSan and LLDB have correct thread numbers.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import json


class TsanThreadNumbersTestCase(TestBase):

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
        self.expect("thread list", "A data race should be detected",
                    substrs=['stopped', 'stop reason = Data race detected'])

        self.assertEqual(
            self.dbg.GetSelectedTarget().process.GetSelectedThread().GetStopReason(),
            lldb.eStopReasonInstrumentation)

        report_thread_id = self.dbg.GetSelectedTarget(
        ).process.GetSelectedThread().GetIndexID()

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
        self.assertEqual(data["issue_type"], "data-race")
        self.assertEqual(len(data["mops"]), 2)

        self.assertEqual(data["mops"][0]["thread_id"], report_thread_id)

        other_thread_id = data["mops"][1]["thread_id"]
        self.assertTrue(other_thread_id != report_thread_id)
        other_thread = self.dbg.GetSelectedTarget(
        ).process.GetThreadByIndexID(other_thread_id)
        self.assertTrue(other_thread.IsValid())

        self.runCmd("thread select %d" % other_thread_id)

        self.expect(
            "thread backtrace",
            "The other thread should be stopped in f1 or f2",
            substrs=[
                "a.out",
                "main.c"])

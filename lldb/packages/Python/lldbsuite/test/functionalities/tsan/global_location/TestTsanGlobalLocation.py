"""
Tests that TSan correctly reports the filename and line number of a racy global variable.
"""

import os
import time
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import json


class TsanGlobalLocationTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="non-core functionality, need to reenable and fix later (DES 2014.11.07)")
    @skipIfFreeBSD  # llvm.org/pr21136 runtimes not yet available by default
    @skipIfRemote
    @skipUnlessCompilerRt
    @skipUnlessThreadSanitizer
    def test(self):
        self.build()
        self.tsan_tests()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def tsan_tests(self):
        exe = os.path.join(os.getcwd(), "a.out")
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

        self.assertTrue(data["location_filename"].endswith("/main.c"))
        self.assertEqual(
            data["location_line"],
            line_number(
                'main.c',
                '// global variable'))

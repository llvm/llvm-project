# TestABIv2.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Test Swift Runtime Reporting ABI v2.
"""
import lldb
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2
import json


class SwiftRuntimeReportingABIv2TestCase(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.skipIfLinux
    def test_swift_runtime_reporting(self):
        self.build()
        self.do_test()

    def setUp(self):
        lldbtest.TestBase.setUp(self)

    def do_test(self):
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        self.runCmd("run")

        self.expect("thread list",
                    substrs=['stopped', 'stop reason = custom error message'])

        self.assertEqual(
            self.dbg.GetSelectedTarget().process.GetSelectedThread().GetStopReason(),
            lldb.eStopReasonInstrumentation)

        self.expect("thread info -s",
            substrs=["instrumentation_class", "issue_type", "description"])

        output_lines = self.res.GetOutput().split('\n')
        json_line = '\n'.join(output_lines[2:])
        data = json.loads(json_line)
        self.assertEqual(data["instrumentation_class"], "SwiftRuntimeReporting")
        self.assertEqual(data["issue_type"], "my-error")
        self.assertEqual(data["memory_address"], 0)
        self.assertEqual(data["description"], "custom error message")

        self.assertEqual(len(data["notes"]), 3)
        self.assertEqual(data["notes"][0]["description"], "note 1")
        self.assertEqual(data["notes"][1]["description"], "note 2")
        self.assertEqual(data["notes"][2]["description"], "note 3")

        self.assertEqual(len(data["notes"][0]["fixits"]), 0)
        self.assertEqual(len(data["notes"][1]["fixits"]), 0)
        self.assertEqual(len(data["notes"][2]["fixits"]), 0)

        self.assertEqual(len(data["fixits"]), 2)
        self.assertEqual(data["fixits"][0]["filename"], "filename1")
        self.assertEqual(data["fixits"][0]["start_line"], 42)
        self.assertEqual(data["fixits"][0]["start_col"], 1)
        self.assertEqual(data["fixits"][0]["end_line"], 43)
        self.assertEqual(data["fixits"][0]["end_col"], 2)
        self.assertEqual(data["fixits"][0]["replacement"], "replacement1")
        self.assertEqual(data["fixits"][1]["filename"], "filename2")
        self.assertEqual(data["fixits"][1]["start_line"], 44)
        self.assertEqual(data["fixits"][1]["start_col"], 3)
        self.assertEqual(data["fixits"][1]["end_line"], 45)
        self.assertEqual(data["fixits"][1]["end_col"], 4)
        self.assertEqual(data["fixits"][1]["replacement"], "replacement2")

# TestObjcInference.py
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
Test support of Swift Runtime Reporting for @objc inference.
"""
import lldb
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2
import json


class SwiftRuntimeReportingObjcInferenceTestCase(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.skipIfLinux
    def test_swift_runtime_reporting(self):
        self.build()
        self.do_test()

    def setUp(self):
        lldbtest.TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        self.runCmd("run")

        self.expect("thread list",
                    substrs=['stopped', 'stop reason = implicit Objective-C entrypoint'])

        thread = target.process.GetSelectedThread()

        self.assertEqual(
            self.dbg.GetSelectedTarget().process.GetSelectedThread().GetStopReason(),
            lldb.eStopReasonInstrumentation)

        self.expect("thread info -s",
            substrs=["instrumentation_class", "issue_type", "description"])

        output_lines = self.res.GetOutput().split('\n')
        json_line = '\n'.join(output_lines[2:])
        data = json.loads(json_line)
        self.assertEqual(data["instrumentation_class"], "SwiftRuntimeReporting")
        self.assertEqual(data["issue_type"], "implicit-objc-entrypoint")
        self.assertEqual(data["description"],
            "implicit Objective-C entrypoint -[a.MyClass memberfunc] is deprecated and will be removed in Swift 4")

        historical_threads = thread.GetStopReasonExtendedBacktraces(lldb.eInstrumentationRuntimeTypeSwiftRuntimeReporting)
        self.assertEqual(historical_threads.GetSize(), 1)

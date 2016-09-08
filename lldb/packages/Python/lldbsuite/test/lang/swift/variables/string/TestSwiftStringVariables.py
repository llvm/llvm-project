# coding=utf-8

# TestSwiftStringVariables.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See http://swift.org/LICENSE.txt for license information
# See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------

"""
Tests simple swift expressions
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftStringVariables(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_swift_string_variables(self):
        """Test that Swift.String formats properly"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Test that Swift.String formats properly"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here', self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        s1 = self.frame.FindVariable("s1")
        s2 = self.frame.FindVariable("s2")

        lldbutil.check_variable(self, s1, summary='"Hello world"')
        lldbutil.check_variable(self, s2, summary='"ΞΕΛΛΘ"')

        TheVeryLongOne = self.frame.FindVariable("TheVeryLongOne")
        summaryOptions = lldb.SBTypeSummaryOptions()
        summaryOptions.SetCapping(lldb.eTypeSummaryUncapped)
        uncappedSummaryStream = lldb.SBStream()
        TheVeryLongOne.GetSummary(uncappedSummaryStream, summaryOptions)
        uncappedSummary = uncappedSummaryStream.GetData()
        self.assertTrue(uncappedSummary.find("someText") > 0,
                        "uncappedSummary does not include the full string")
        summaryOptions.SetCapping(lldb.eTypeSummaryCapped)
        cappedSummaryStream = lldb.SBStream()
        TheVeryLongOne.GetSummary(cappedSummaryStream, summaryOptions)
        cappedSummary = cappedSummaryStream.GetData()
        self.assertTrue(
            cappedSummary.find("someText") <= 0,
            "cappedSummary includes the full string")
        self.assertTrue(
            cappedSummary.endswith('"...'),
            "cappedSummary ends with quote dot dot dot")

        IContainZerosASCII = self.frame.FindVariable("IContainZerosASCII")
        IContainZerosUnicode = self.frame.FindVariable("IContainZerosUnicode")
        IContainEscapes = self.frame.FindVariable("IContainEscapes")

        lldbutil.check_variable(
            self,
            IContainZerosASCII,
            summary='"a\\0b\\0c\\0d"')
        lldbutil.check_variable(
            self,
            IContainZerosUnicode,
            summary='"HFIHЗIHF\\0VЭHVHЗ90HGЭ"')
        lldbutil.check_variable(
            self,
            IContainEscapes,
            summary='"Hello\\u{8}\\n\\u{8}\\u{8}\\nGoodbye"')

        self.expect(
            'expression -l objc++ -- (char*)"Hello\b\b\b\b\bGoodbye"',
            substrs=['"Hello\\b\\b\\b\\b\\bGoodbye"'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

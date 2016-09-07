# coding=utf8
"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxStringDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    @skipIf(compiler="gcc")
    @skipIfWindows  # libc++ not ported to Windows yet
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1)

        self.runCmd("run", RUN_SUCCEEDED)

        lldbutil.skip_if_library_missing(
            self, self.target(), lldbutil.PrintableRegex("libc\+\+"))

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)
            self.runCmd(
                "settings set target.max-children-count 256",
                check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.expect(
            "frame variable",
            substrs=[
                '(std::__1::wstring) s = L"hello world! מזל טוב!"',
                '(std::__1::wstring) S = L"!!!!"',
                '(const wchar_t *) mazeltov = 0x',
                'L"מזל טוב"',
                '(std::__1::string) q = "hello world"',
                '(std::__1::string) Q = "quite a long std::strin with lots of info inside it"',
                '(std::__1::string) IHaveEmbeddedZeros = "a\\0b\\0c\\0d"',
                '(std::__1::wstring) IHaveEmbeddedZerosToo = L"hello world!\\0てざ ル゜䋨ミ㠧槊 きゅへ狦穤襩 じゃ馩リョ 䤦監"'])

        self.runCmd("n")

        TheVeryLongOne = self.frame().FindVariable("TheVeryLongOne")
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

        self.expect(
            "frame variable",
            substrs=[
                '(std::__1::wstring) s = L"hello world! מזל טוב!"',
                '(std::__1::wstring) S = L"!!!!!"',
                '(const wchar_t *) mazeltov = 0x',
                'L"מזל טוב"',
                '(std::__1::string) q = "hello world"',
                '(std::__1::string) Q = "quite a long std::strin with lots of info inside it"',
                '(std::__1::string) IHaveEmbeddedZeros = "a\\0b\\0c\\0d"',
                '(std::__1::wstring) IHaveEmbeddedZerosToo = L"hello world!\\0てざ ル゜䋨ミ㠧槊 きゅへ狦穤襩 じゃ馩リョ 䤦監"'])

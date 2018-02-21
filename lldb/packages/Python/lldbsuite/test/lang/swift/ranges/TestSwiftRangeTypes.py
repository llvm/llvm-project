# TestSwiftRangeTypes.py
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
Test the Swift.Range<T> type
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftRangeType(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.add_test_categories(["swiftpr"])
    def test_swift_range_type(self):
        """Test the Swift.Range<T> type"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Test the Swift.Range<T> type"""

        (target, process, self.thread, breakpoint) = lldbutil.run_to_source_breakpoint(
                self, 'Set breakpoint here', self.main_source_spec)
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        self.expect("frame variable a", substrs=[
                    '(ClosedRange<Int>) a = 1...100'])
        self.expect("frame variable b", substrs=['(Range<Int>) b = 1..<100'])
        self.expect("frame variable c", substrs=[
                    '(ClosedRange<Int>) c = 1...100'])
        self.expect("frame variable d", substrs=[
                    '(Range<Int>) d = 1..<100'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

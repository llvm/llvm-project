# TestSwiftValueOfOptionals.py
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
Check that trying to read an optional's numeric value doesn't crash LLDB
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftValueOfOptionalType(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.add_test_categories(["swiftpr"])
    def test_swift_value_optional_type(self):
        """Check that trying to read an optional's numeric value doesn't crash LLDB"""
        self.build()
        self.do_check()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_check(self):
        """Check that trying to read an optional's numeric value doesn't crash LLDB"""
        s = self.frame().FindVariable("s")
        self.assertTrue(s.GetValueAsSigned(0) == 0, "reading value fails")
        self.assertTrue(s.GetValueAsSigned(1) == 1, "reading value fails")
        self.assertTrue(s.GetValueAsUnsigned(0) == 0, "reading value fails")
        self.assertTrue(s.GetValueAsUnsigned(1) == 1, "reading value fails")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

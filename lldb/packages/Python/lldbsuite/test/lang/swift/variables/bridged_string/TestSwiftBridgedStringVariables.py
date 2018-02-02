# coding=utf-8

# TestSwiftBridgedStringVariables.py
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
Tests simple swift expressions
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftBridgedStringVariables(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    @decorators.add_test_categories(["swiftpr"])
    def test_swift_bridged_string_variables(self):
        """Test that Swift.String formats properly"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Test that Swift.String formats properly"""
        (_, _, thread, _) = lldbutil.run_to_source_breakpoint(self,
                "Set breakpoint here", self.main_source_spec)
        self.frame = thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        s1 = self.frame.FindVariable("s1")
        s2 = self.frame.FindVariable("s2")
        s3 = self.frame.FindVariable("s3")
        s4 = self.frame.FindVariable("s4")
        s5 = self.frame.FindVariable("s5")
        s6 = self.frame.FindVariable("s6")

        lldbutil.check_variable(self, s1, summary='"Hello world"')
        lldbutil.check_variable(self, s2, summary='"ΞΕΛΛΘ"')
        lldbutil.check_variable(self, s3, summary='"Hello world"')
        lldbutil.check_variable(self, s4, summary='"ΞΕΛΛΘ"')
        lldbutil.check_variable(self, s5, use_dynamic=True, summary='"abc"')
        lldbutil.check_variable(self, s6, summary='"abc"')

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

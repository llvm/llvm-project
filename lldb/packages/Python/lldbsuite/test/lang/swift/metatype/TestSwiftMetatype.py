# TestSwiftMetatype.py
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
Test the formatting of Swift metatypes
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftMetatype(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    @decorators.swiftTest
    @decorators.add_test_categories(["swiftpr"])
    def test_metatype(self):
        """Test the formatting of Swift metatypes"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        frame = thread.frames[0]
        self.assertTrue(frame, "Frame 0 is valid.")

        var_s = frame.FindVariable("s")
        var_c = frame.FindVariable("c")
        var_f = frame.FindVariable("f")
        var_t = frame.FindVariable("t")
        var_p = frame.FindVariable("p")
        lldbutil.check_variable(self, var_s, False, "String")
        lldbutil.check_variable(self, var_c, False, "a.D")
        lldbutil.check_variable(self, var_f, False, "(Int) -> Int")
        lldbutil.check_variable(self, var_t, False, "(Int, Int, String)")
        lldbutil.check_variable(self, var_p, False, "P")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

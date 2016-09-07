# TestNSError.py
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
Tests that Swift displays NSError correctly
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class SwiftNSErrorTest(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    @decorators.expectedFailureAll(
        bugnumber="https://bugs.swift.org/browse/SR-782")
    def test_swift_nserror(self):
        """Tests that Swift displays NSError correctly"""
        self.build()
        self.nserror_commands()

    def setUp(self):
        TestBase.setUp(self)

    def nserror_commands(self):
        """Tests that Swift displays NSError correctly"""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_source_regexp(
            self, "// Set a breakpoint here")

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs=['stopped', 'stop reason = breakpoint'])

        self.expect("frame variable -d run --ptr-depth=2", substrs=[
            '0 = " "', '1 = 0x', 'domain: "lldbrocks" - code: 3133079277 {',
            '_userInfo = ', '2 key/value pairs {',
            '0 = ', ' "x"', '1 = ', ' Int64(0)', '0 = ', ' "y"', '1 = ',
            ' Int64(0)', '0 = "x+y"', 'domain: "lldbrocks" - code: 0 {',
            '1 = ', ' Int64(3)', '1 = ', ' Int64(4)'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

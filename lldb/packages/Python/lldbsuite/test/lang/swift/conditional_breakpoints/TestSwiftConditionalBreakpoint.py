# TestSwiftConditionalBreakpoint.py
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
Tests that we can set a conditional breakpoint in Swift code
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftConditionalBreakpoint(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.skipIfLinux
    def test_swift_conditional_breakpoint(self):
        """Tests that we can set a conditional breakpoint in Swift code"""
        self.build()
        self.break_commands()

    def setUp(self):
        TestBase.setUp(self)

    def break_commands(self):
        """Tests that we can set a conditional breakpoint in Swift code"""
        exe_name = self.getBuildArtifact("a.out")
        self.runCmd("file %s"%(exe_name), CURRENT_EXECUTABLE_SET)
        bkno = lldbutil.run_break_set_by_source_regexp(
            self, "Set breakpoint here")
        self.runCmd('breakpoint modify ' + str(bkno) + ' -c x==y')

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        self.expect("frame var x y", substrs=['x = 5', 'y = 5'])

        self.runCmd("continue", RUN_SUCCEEDED)

        self.expect("frame var x y", substrs=['x = 6', 'y = 6'])

        self.runCmd('breakpoint modify ' + str(bkno) + ' -c x>y')

        self.runCmd("continue", RUN_SUCCEEDED)

        self.expect("frame var x y", substrs=['x = 3', 'y = 1'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

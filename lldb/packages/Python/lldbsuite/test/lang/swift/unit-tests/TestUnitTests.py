# TestUnitTests.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See http://swift.org/LICENSE.txt for license information
# See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Test that XCTest-based unit tests work
"""
import commands
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import unittest2


def execute_command(command):
    (exit_status, output) = commands.getstatusoutput(command)
    return exit_status


class TestUnitTests(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    @decorators.skipIf(
        debug_info=decorators.no_match("dsym"),
        bugnumber="This test only builds one way")
    def test_cross_module_extension(self):
        """Test that XCTest-based unit tests work"""
        self.buildAll()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.XCTest_source = "XCTest.c"
        self.XCTest_source_spec = lldb.SBFileSpec(self.XCTest_source)

    def buildAll(self):
        execute_command("make everything")

    def do_test(self):
        """Test that XCTest-based unit tests work"""
        exe_name = "XCTest"
        exe = os.path.join(os.getcwd(), exe_name)

        def cleanup():
            execute_command("make cleanup")
        self.addTearDownHook(cleanup)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here', self.XCTest_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        options = lldb.SBExpressionOptions()
        options.SetLanguage(lldb.eLanguageTypeSwift)

        self.frame.EvaluateExpression("import test", options)

        ret = self.frame.EvaluateExpression("doTest()", options)

        self.assertTrue(ret.GetValueAsUnsigned() == 3)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

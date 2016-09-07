# TestSwiftDifferentClangFlags.py
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
Test that we use the right compiler flags when debugging
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
    # print '%% %s' % (command)
    (exit_status, output) = commands.getstatusoutput(command)
    # if output:
    #     print output
    # print 'status = %u' % (exit_status)
    return exit_status


class TestSwiftDifferentClangFlags(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    @decorators.skipIf(
        debug_info=decorators.no_match("dsym"),
        bugnumber="This test requires a stripped binary and a dSYM")
    @decorators.skipIf(oslist=["macosx"], bugnumber="rdar://26051347")
    def test_swift_different_clang_flags(self):
        """Test that we use the right compiler flags when debugging"""
        self.buildAll()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        self.modb_source = "modb.swift"
        self.modb_source_spec = lldb.SBFileSpec(self.modb_source)

    def buildAll(self):
        execute_command("make everything")

    def do_test(self):
        """Test that we use the right compiler flags when debugging"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        def cleanup():
            execute_command("make cleanup")
        self.addTearDownHook(cleanup)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        main_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', self.main_source_spec)
        self.assertTrue(
            main_breakpoint.GetNumLocations() > 0,
            VALID_BREAKPOINT)

        modb_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', self.modb_source_spec)
        self.assertTrue(
            modb_breakpoint.GetNumLocations() > 0,
            VALID_BREAKPOINT)

        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, modb_breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        var = self.frame.FindVariable("myThree")
        three = var.GetChildMemberWithName("three")
        lldbutil.check_variable(self, var, False, typename="modb.MyStruct")
        lldbutil.check_variable(self, three, False, value="3")

        process.Continue()
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, main_breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        var = self.frame.FindVariable("a")
        lldbutil.check_variable(self, var, False, value="2")
        var = self.frame.FindVariable("b")
        lldbutil.check_variable(self, var, False, value="3")

        var = self.frame.EvaluateExpression("fA()")
        lldbutil.check_variable(self, var, False, value="2")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

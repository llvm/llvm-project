# TestSwiftCrossModuleExtension.py
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
Test that we correctly find private extension decls across modules
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


class TestSwiftCrossModuleExtension(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    def test_cross_module_extension(self):
        """Test that we correctly find private extension decls across modules"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.a_source = "moda.swift"
        self.a_source_spec = lldb.SBFileSpec(self.a_source)
        self.b_source = "modb.swift"
        self.b_source_spec = lldb.SBFileSpec(self.b_source)

    def do_test(self):
        """Test that we correctly find private extension decls across modules"""
        exe_name = "main"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        a_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', self.a_source_spec)
        self.assertTrue(a_breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)
        b_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', self.b_source_spec)
        self.assertTrue(b_breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, a_breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        var = self.frame.FindVariable("a")
        child_v = var.GetChildMemberWithName("v")
        lldbutil.check_variable(self, var, False, typename="moda.S.A")
        lldbutil.check_variable(self, child_v, False, value="1")

        process.Continue()
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, b_breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        var = self.frame.FindVariable("a")
        child_v = var.GetChildMemberWithName("v")
        lldbutil.check_variable(self, var, False, typename="moda.S.A")
        lldbutil.check_variable(self, child_v, False, value="3")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

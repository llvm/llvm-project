# TestSwiftPrivateTypeAlias.py
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
Test that we correctly find private decls
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftPrivateTypeAlias(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.a_source = "main.swift"
        self.a_source_spec = lldb.SBFileSpec(self.a_source)

    @decorators.swiftTest
    @decorators.expectedFailureAll(bugnumber="rdar://24921067")
    def test_swift_private_typealias(self):
        """Test that we can correctly print variables whose types are private type aliases"""
        self.build()
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint1 = target.BreakpointCreateBySourceRegex(
            'breakpoint 1', self.a_source_spec)
        breakpoint2 = target.BreakpointCreateBySourceRegex(
            'breakpoint 2', self.a_source_spec)
        self.assertTrue(breakpoint1.GetNumLocations() > 0, VALID_BREAKPOINT)
        self.assertTrue(breakpoint2.GetNumLocations() > 0, VALID_BREAKPOINT)

        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint1)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        var = self.frame.FindVariable("i")
        lldbutil.check_variable(
            self,
            var,
            False,
            typename="a.MyStruct.Type.IntegerType",
            value="123")

        process.Continue()
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint2)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        var = self.frame.FindVariable("a")
        dict_child_0 = var.GetChildAtIndex(0)
        child_0 = dict_child_0.GetChildAtIndex(0)
        child_1 = dict_child_0.GetChildAtIndex(1)
        lldbutil.check_variable(
            self,
            var,
            False,
            typename="Swift.Dictionary<Swift.String, a.MyStruct.Type.IntegerType>")
        lldbutil.check_variable(self, child_0, False, '"hello"')
        lldbutil.check_variable(self, child_1, False, value='234')

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

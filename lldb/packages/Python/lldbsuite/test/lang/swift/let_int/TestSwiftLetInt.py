# TestSwiftLetInt.py
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
Test that a 'let' Int is formatted properly
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftLetIntSupport(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.add_test_categories(["swiftpr"])
    def test_swift_let_int(self):
        """Test that a 'let' Int is formatted properly"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Test that a 'let' Int is formatted properly"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here', self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        let = self.frame.FindVariable("x")
        var = self.frame.FindVariable("y")
        lldbutil.check_variable(self, let, False, value="10")
        lldbutil.check_variable(self, var, False, value="10")

        get_arguments = False
        get_locals = True
        get_statics = False
        get_in_scope_only = True
        local_vars = self.frame.GetVariables(get_arguments, get_locals,
                                             get_statics, get_in_scope_only)
        self.assertTrue(local_vars.GetFirstValueByName("x").IsValid())
        self.assertTrue(local_vars.GetFirstValueByName("y").IsValid())
        self.assertTrue(not local_vars.GetFirstValueByName("z").IsValid())

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

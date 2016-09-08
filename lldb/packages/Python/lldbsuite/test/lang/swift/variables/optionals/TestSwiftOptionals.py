# TestSwiftOptionals.py
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
Check formatting for T? and T!
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftOptionalType(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_swift_optional_type(self):
        """Check formatting for T? and T!"""
        self.build()
        self.do_check_consistency()
        self.do_check_visuals()
        self.do_check_api()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_check_consistency(self):
        """Check formatting for T? and T!"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

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

    def do_check_visuals(self):
        """Check formatting for T? and T!"""
        self.expect(
            "frame variable optS_Some",
            substrs=[
                'a = 12',
                'b = "Hello world"'])
        self.expect(
            "frame variable uoptS_Some",
            substrs=[
                'a = 12',
                'b = "Hello world"'])

        self.expect("frame variable optString_Some", substrs=['"hello"'])
        self.expect("frame variable uoptString_Some", substrs=['"hello"'])

        self.expect("frame variable optS_None", substrs=['nil'])
        self.expect("frame variable uoptS_None", substrs=['nil'])

        self.expect("frame variable optString_None", substrs=['nil'])
        self.expect("frame variable uoptString_None", substrs=['nil'])

    def do_check_api(self):
        """Check formatting for T? and T!"""
        optS_Some = self.frame.FindVariable("optS_Some")
        lldbutil.check_variable(
            self,
            optS_Some,
            use_dynamic=False,
            num_children=2)
        uoptS_Some = self.frame.FindVariable("uoptS_Some")
        lldbutil.check_variable(
            self,
            uoptS_Some,
            use_dynamic=False,
            num_children=2)

        optString_None = self.frame.FindVariable("optString_None")
        lldbutil.check_variable(
            self,
            optString_None,
            use_dynamic=False,
            num_children=0)
        uoptString_None = self.frame.FindVariable("uoptString_None")
        lldbutil.check_variable(
            self,
            uoptString_None,
            use_dynamic=False,
            num_children=0)

        optString_Some = self.frame.FindVariable("optString_Some")
        lldbutil.check_variable(
            self,
            optString_Some,
            use_dynamic=False,
            num_children=1)
        uoptString_Some = self.frame.FindVariable("uoptString_Some")
        lldbutil.check_variable(
            self,
            uoptString_Some,
            use_dynamic=False,
            num_children=1)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

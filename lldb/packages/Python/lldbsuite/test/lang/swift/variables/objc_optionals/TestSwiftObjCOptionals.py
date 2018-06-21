# TestSwiftObjCOptionals.py
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
Check formatting for T? and T! when T is an ObjC type
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftObjCOptionalType(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.skipUnlessDarwin
    @decorators.add_test_categories(["swiftpr"])
    def test_swift_objc_optional_type(self):
        """Check formatting for T? and T! when T is an ObjC type"""
        self.build()
        self.do_check_consistency()
        self.do_check_visuals()
        self.do_check_api()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_check_consistency(self):
        """Check formatting for T? and T! when T is an ObjC type"""
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

    def do_check_visuals(self):
        """Check formatting for T? and T! when T is an ObjC type"""
        self.expect(
            "frame variable optColor_Some",
            substrs=['(Color?) optColor_Some = 0x'])
        self.expect(
            "frame variable uoptColor_Some",
            substrs=['(Color?) uoptColor_Some = 0x'])

        self.expect("frame variable optColor_None", substrs=['nil'])
        self.expect("frame variable uoptColor_None", substrs=['nil'])

    def do_check_api(self):
        """Check formatting for T? and T! when T is an ObjC type"""
        optColor_Some = self.frame.FindVariable("optColor_Some")
        lldbutil.check_variable(
            self,
            optColor_Some,
            use_dynamic=False,
            num_children=1)
        uoptColor_Some = self.frame.FindVariable("uoptColor_Some")
        lldbutil.check_variable(
            self,
            uoptColor_Some,
            use_dynamic=False,
            num_children=1)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

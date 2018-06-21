# TestSwiftStructChangeRerun.py
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
Test that we display self correctly for an inline-initialized struct
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import shutil
import unittest2


class TestSwiftStructChangeRerun(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.skipIf(debug_info=decorators.no_match("dsym"))
    @decorators.swiftTest
    def test_swift_struct_change_rerun(self):
        """Test that we display self correctly for an inline-initialized struct"""
        self.do_test(True)

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self, build_dsym):
        """Test that we display self correctly for an inline-initialized struct"""

        # Cleanup the copied source file
        def cleanup():
            if os.path.exists(self.main_source):
                os.unlink(self.main_source)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        if os.path.exists(self.main_source):
            os.unlink(self.main_source)
        shutil.copyfile('main1.swift', self.main_source)
        print 'build with main1.swift'
        self.build()
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

        var_a = self.frame.EvaluateExpression("a")
        print var_a
        var_a_a = var_a.GetChildMemberWithName("a")
        lldbutil.check_variable(self, var_a_a, False, value="12")

        var_a_b = var_a.GetChildMemberWithName("b")
        lldbutil.check_variable(self, var_a_b, False, '"Hey"')

        var_a_c = var_a.GetChildMemberWithName("c")
        self.assertFalse(var_a_c.IsValid(), "make sure a.c doesn't exist")

        process.Kill()

        print 'build with main2.swift'
        os.unlink(self.main_source)
        shutil.copyfile('main2.swift', self.main_source)
        if build_dsym:
            self.buildDsym()
        else:
            self.buildDwarf()

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

        var_a = self.frame.EvaluateExpression("a")
        print var_a
        var_a_a = var_a.GetChildMemberWithName("a")
        lldbutil.check_variable(self, var_a_a, False, value="12")

        var_a_b = var_a.GetChildMemberWithName("b")
        lldbutil.check_variable(self, var_a_b, False, '"Hey"')

        var_a_c = var_a.GetChildMemberWithName("c")
        self.assertTrue(var_a_c.IsValid(), "make sure a.c does exist")
        lldbutil.check_variable(self, var_a_c, False, value='12.125')

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

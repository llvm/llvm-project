# TestSwiftHideRuntimeSupport.py
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
Test that we hide runtime support values
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftHideRuntimeSupport(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_swift_hide_runtime_support(self):
        """Test that we hide runtime support values"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Test that we hide runtime support values"""

        # This is the function to remove the custom settings in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd(
                'settings set target.display-runtime-support-values true',
                check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("settings set target.display-runtime-support-values false")

        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', self.main_source_spec)
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

        self.expect(
            'frame variable -d run',
            substrs=['_0_0'],
            matching=False)
        self.expect('frame variable -d run', substrs=['193627'], matching=True)

        var_opts = lldb.SBVariablesOptions()
        var_opts.SetIncludeArguments(True)
        var_opts.SetIncludeLocals(True)
        var_opts.SetInScopeOnly(True)
        var_opts.SetIncludeStatics(True)
        var_opts.SetIncludeRuntimeSupportValues(False)
        var_opts.SetUseDynamic(lldb.eDynamicCanRunTarget)

        values = self.frame.GetVariables(var_opts)
        found = False
        for value in values:
            if '_0_0' in value.name:
                found = True
        self.assertFalse(found, "found the thing I was not expecting")

        var_opts.SetIncludeRuntimeSupportValues(True)
        values = self.frame.GetVariables(var_opts)
        found = False
        for value in values:
            if '_0_0' in value.name:
                found = True
        self.assertTrue(found, "not found the thing I was expecting")

        self.runCmd("settings set target.display-runtime-support-values true")
        self.expect(
            'frame variable -d run',
            substrs=['_0_0'],
            matching=True)

        self.runCmd("settings set target.display-runtime-support-values false")
        self.expect(
            'frame variable -d run',
            substrs=['_0_0'],
            matching=False)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

# TestEnumVariables.py
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
Tests that Enum variables display correctly
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestEnumVariables(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.add_test_categories(["swiftpr"])
    def test_enum_variables(self):
        """Tests that Enum variables display correctly"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def get_variable(self, name):
        return self.frame.FindVariable(
            name).GetDynamicValue(lldb.eDynamicCanRunTarget)

    def check_enum(
            self,
            var_name,
            value,
            child_summary=None,
            child_value=None):
        var = self.frame.FindVariable(var_name)
        self.assertTrue(var.IsValid(), "invalid variable")
        self.assertTrue(var.GetValue() == value, "invalid value")
        if child_summary:
            child_var = var.GetChildMemberWithName(value)
            self.assertTrue(child_var.IsValid(), "invalid child")
            if child_summary:
                self.assertTrue(
                    child_var.GetSummary() == child_summary,
                    "invalid child summary")
            if child_value:
                self.assertTrue(
                    child_var.GetValue() == child_value,
                    "invalid child value")

    def do_test(self):
        """Tests that Enum variables display correctly"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            '// Set breakpoint here', self.main_source_spec)
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

        #self.runCmd("frame variable")

        self.check_enum("ona", "A")
        self.check_enum("onb", "B")
        self.check_enum("onc", "C")
        self.check_enum("ond", "D")

        self.check_enum("twa", "A")
        self.check_enum("twb", "B", '"hello world"')
        self.check_enum("twc", "C", child_value='12')
        self.check_enum("twd", "D")

        self.check_enum("tha", "A", '"hello world"')
        self.check_enum("thb", "B", child_value='24')
        self.check_enum("thc", "C", '"this is me"')
        self.check_enum("thd", "D", "true")

        self.check_enum("foa", "A", '"hello world"')
        self.check_enum("fob", "B", '"this is me"')
        self.check_enum("foc", "C", '"life should be"')
        self.check_enum("fod", "D", '"fun for everyone"')

        self.expect('frame variable ContainerOfEnums_Some',
                    substrs=['Some', 'one1 = A', 'one2 = A'])
        self.expect('frame variable ContainerOfEnums_Nil', substrs=['nil'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

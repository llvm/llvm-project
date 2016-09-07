# TestSwiftGenericEnums.py
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
Test that we handle reasonably generically-typed enums
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftGenericEnumTypes(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.expectedFailureAll(bugnumber="Pending investigation")
    def test_swift_generic_enum_types(self):
        """Test that we handle reasonably generically-typed enums"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def get_variable(self, name):
        var = self.frame.FindVariable(
            name).GetDynamicValue(lldb.eDynamicCanRunTarget)
        var.SetPreferSyntheticValue(True)
        return var

    # using Ints as keys should practically guarantee that key and index match
    # that makes the test case logic much much easier
    def check_dictionary_entry(self, var, index, key_summary, value_summary):
        self.assertTrue(var.GetChildAtIndex(index).IsValid(), "invalid item")
        self.assertTrue(var.GetChildAtIndex(index).GetChildMemberWithName(
            "key").IsValid(), "invalid key child")
        self.assertTrue(var.GetChildAtIndex(index).GetChildMemberWithName(
            "value").IsValid(), "invalid key child")
        self.assertTrue(var.GetChildAtIndex(index).GetChildMemberWithName(
            "key").GetSummary() == key_summary, "invalid key summary")
        self.assertTrue(var.GetChildAtIndex(index).GetChildMemberWithName(
            "value").GetSummary() == value_summary, "invalid value summary")

    def keep_going(self, process, breakpoint):
        process.Continue()
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

    def do_test(self):
        """Tests that we properly vend synthetic children for Swift.Dictionary"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint1 = target.BreakpointCreateBySourceRegex(
            '// Set first breakpoint here.', self.main_source_spec)
        breakpoint2 = target.BreakpointCreateBySourceRegex(
            '// Set second breakpoint here.', self.main_source_spec)
        self.assertTrue(breakpoint1.GetNumLocations() > 0, VALID_BREAKPOINT)
        self.assertTrue(breakpoint2.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint1)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd("type summary delete main.StringWrapper", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        enumvar = self.get_variable("myOptionalU").GetStaticValue()
        self.assertTrue(enumvar.GetValue() is None,
                        "static type has a value when it shouldn't")
        enumvar = enumvar.GetDynamicValue(lldb.eDynamicCanRunTarget)
        self.assertTrue(
            enumvar.GetValue() == "Some",
            "dynamic type's value should be Some")
        self.assertTrue(
            enumvar.GetSummary() == "3",
            "Some's summary should be 3")

        self.runCmd("continue")
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint2)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        value = self.get_variable("value")
        lldbutil.check_variable(
            self,
            value,
            use_dynamic=True,
            summary='"Now with Content"',
            typename='String?')

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

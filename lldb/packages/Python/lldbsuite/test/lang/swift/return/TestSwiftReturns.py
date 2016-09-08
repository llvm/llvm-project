# TestSwiftReturns.py
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
Test getting return values
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftReturns(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.skipIfLinux  # bugs.swift.org/SR-841
    @decorators.expectedFailureDarwin("rdar://problem/25073875")
    def test_swift_returns(self):
        """Test getting return values"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def step_out_until_no_breakpoint(self, breakpoint, continue_process):
        if continue_process:
            self.process.Continue()
        # Step out and make sure after we step out we are not stopped at a
        # breakpoint
        while True:
            threads = lldbutil.get_threads_stopped_at_breakpoint(
                self.process, breakpoint)
            if len(threads) == 0:
                break
            self.thread.StepOut()

    def compare_value(self, a, b, compare_name):
        '''Compare to lldb.SBValue objects "a" and "b"

        Returns an error string describing what parts of "a" and "b" didn't match
        or returns None for a successful compare'''
        if compare_name and a.name != b.name:
            return 'error: name mismatch "%s" != "%s"' % (a.name, b.name)
        if a.type.name != b.type.name:
            return 'error: typename mismatch "%s" != "%s"' % (
                a.type.name, b.type.name)
        if a.value != b.value:
            return 'error: value string mismatch "%s" != "%s"' % (
                a.value, b.value)
        if a.summary != b.summary:
            return 'error: summary mismatch "%s" != "%s"' % (
                a.summary, b.summary)
        if a.num_children != b.num_children:
            return 'error: num_children mismatch %i != %i' % (
                a.num_children, b.num_children)
        for i in range(a.num_children):
            a_child = a.GetChildAtIndex(i, lldb.eNoDynamicValues, False)
            b_child = b.GetChildAtIndex(i, lldb.eNoDynamicValues, False)
            err = self.compare_value(a_child, b_child, True)
            if err:
                return err
        return None  # Success if we return None

    def verify_return_value_against_local_variable(
            self, return_value, variable_name):
        frame = self.thread.frame[0]
        variable = frame.FindVariable(variable_name, lldb.eDynamicCanRunTarget)
        self.assertTrue(variable)
        variable.SetPreferSyntheticValue(True)

        # Don't compare the names as the return value name won't match the
        # variable name
        err = self.compare_value(return_value, variable, False)
        if err:
            if self.TraceOn():
                print 'return value: %s' % (return_value)
                print '    variable: %s' % (variable)
            self.assertTrue(False, err)

    def verify_invisibility(self, variables):
        frame = self.thread.frame[0]
        for var_name in variables:
            variable = frame.FindVariable(var_name, lldb.eDynamicCanRunTarget)
            self.assertTrue(variable)
            self.assertTrue(variable.value is None)

    def do_test(self):
        """Tests that we can break and display simple types"""
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
        self.process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(self.process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            self.process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]

        line_before_fin = self.thread.frame[1].line_entry.line
        self.step_out_until_no_breakpoint(breakpoint, False)

        # Get a "Swift.Int64" return struct value
        # Get a "main.Foo" return class value
        # Get a "Swift.String" return class value
        # Get a "Swift.Dictionary<Swift.Int, Swift.String>" return class value
        # Get a "Swift.String?" return class value
        # Get a "Swift.Float" return class value
        # Get a "Swift.Double" return class value
        # Test that the local variable equals return value.
        # Test that none of the later let-bound values are available.
        variables = ["u", "i", "c", "s", "dict", "opt_str", "f", "d"]
        # FIXME: Enable opt_str below:
        invisibles = ["i", "c", "s", "dict", "f", "f", "d"]
        for var in variables:
            return_value = self.thread.GetStopReturnValue()
            if line_before_fin == self.thread.frame[0].line_entry.line:
                self.thread.StepOver()
            self.verify_return_value_against_local_variable(return_value, var)
            self.verify_invisibility(invisibles)
            if len(invisibles):
                invisibles.pop(0)
            line_before_fin = self.thread.frame[0].line_entry.line
            self.step_out_until_no_breakpoint(breakpoint, True)

        # Call a function that could throw but doesn't and see that it actually
        # gets the result:
        return_value = self.thread.GetStopReturnValue()
        if line_before_fin == self.thread.frame[0].line_entry.line:
            self.thread.StepOver()
        self.verify_return_value_against_local_variable(
            return_value, "not_err")

        # Call it again, but this time it will throw:
        # The return will be empty, but the error will not be:
        line_before_fin = self.thread.frame[0].line_entry.line
        self.step_out_until_no_breakpoint(breakpoint, True)
        return_value = self.thread.GetStopReturnValue()
        self.assertTrue(not return_value.IsValid())
        error_value = self.thread.GetStopErrorValue()
        if line_before_fin == self.thread.frame[0].line_entry.line:
            self.thread.StepOver()
        self.verify_return_value_against_local_variable(error_value, "err")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

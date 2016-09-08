# TestExpressionScopes.py
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
Tests scoped variables with swift expressions
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftExpressionScopes(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_expression_scopes(self):
        """Tests that swift expressions resolve scoped variables correctly"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def check_expression(self, expression, expected_result, use_summary=True):
        value = self.frame.EvaluateExpression(expression)
        self.assertTrue(value.IsValid(), expression + "returned a valid value")
        if use_summary:
            answer = value.GetSummary()
        else:
            answer = value.GetValue()
        report_str = "%s expected: %s got: %s" % (
            expression, expected_result, answer)
        if answer != expected_result:
            print report_str
            print value.GetError()

        self.assertTrue(answer == expected_result, report_str)

    def continue_to_bkpt(self, process, bkpt):
        threads = lldbutil.continue_to_breakpoint(process, bkpt)
        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

    def continue_by_pattern(self, pattern):
        bkpt = self.target.BreakpointCreateBySourceRegex(
            pattern, self.main_source_spec)
        self.assertTrue(bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)
        self.continue_to_bkpt(self.process, bkpt)
        self.target.BreakpointDelete(bkpt.GetID())

    def do_test(self):
        """Tests that swift expressions resolve scoped variables correctly"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.target = target
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        shadow_a_bkpt = target.BreakpointCreateBySourceRegex(
            'Shadowed in A', self.main_source_spec)
        self.assertTrue(shadow_a_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        init_bkpt = target.BreakpointCreateBySourceRegex(
            'In init.', self.main_source_spec)
        self.assertTrue(init_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.process = process

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, init_bkpt)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        self.check_expression("in_class_a", "20", use_summary=False)
        self.check_expression("self.in_class_a", "20", use_summary=False)

        # Disable the init breakpoint so we don't have to hit it again:
        init_bkpt.SetEnabled(False)

        # Now continue to shadowed_in_a:
        self.continue_to_bkpt(process, shadow_a_bkpt)

        self.check_expression("in_class_a", "10", use_summary=False)
        self.check_expression("self.in_class_a", "20", use_summary=False)
        self.check_expression("self.also_in_a", "21", use_summary=False)
        self.check_expression("also_in_a", "21", use_summary=False)

        shadow_a_bkpt.SetEnabled(False)

        # Now set a breakpoint in the static_method and run to there:
        static_bkpt = target.BreakpointCreateBySourceRegex(
            'In class function', self.main_source_spec)
        self.assertTrue(static_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        self.continue_to_bkpt(process, static_bkpt)

        self.check_expression("input", "10", use_summary=False)
        # This test fails on Bryce.  The self metatype doesn't generate a valid
        # type.
        self.check_expression("self.return_ten()", "10", use_summary=False)
        static_bkpt.SetEnabled(False)

        # Now run into the setters & getters:
        set_bkpt = target.BreakpointCreateBySourceRegex(
            'In set.', self.main_source_spec)
        self.assertTrue(set_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        self.continue_to_bkpt(process, set_bkpt)
        self.check_expression("self.backing_int", "10", use_summary=False)
        set_bkpt.SetEnabled(False)

        get_bkpt = target.BreakpointCreateBySourceRegex(
            'In get.', self.main_source_spec)
        self.assertTrue(get_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        self.continue_to_bkpt(process, get_bkpt)
        self.check_expression("self.backing_int", "41", use_summary=False)
        get_bkpt.SetEnabled(False)

        deinit_bkpt = target.BreakpointCreateBySourceRegex(
            'In deinit.', self.main_source_spec)
        self.assertTrue(deinit_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        self.continue_to_bkpt(process, deinit_bkpt)
        self.check_expression("self.backing_int", "41", use_summary=False)
        deinit_bkpt.SetEnabled(False)

        # Now let's try the subscript getter & make sure that that works:

        str_sub_get_bkpt = target.BreakpointCreateBySourceRegex(
            'In string subscript getter', self.main_source_spec)
        self.assertTrue(
            str_sub_get_bkpt.GetNumLocations() > 0,
            VALID_BREAKPOINT)

        self.continue_to_bkpt(process, str_sub_get_bkpt)
        self.check_expression("self.backing_int", "10", use_summary=False)

        str_sub_get_bkpt.SetEnabled(False)

        # Next run into the closure that captures self and make sure that
        # works:
        closure_bkpt = target.BreakpointCreateBySourceRegex(
            'Break here in closure', self.main_source_spec)
        self.assertTrue(closure_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        self.continue_to_bkpt(process, closure_bkpt)
        self.check_expression("self.backing_int", "10", use_summary=False)
        self.check_expression("a_string", '"abcde"', use_summary=True)

        # Now set a breakpoint in the struct method and run to there:
        self.continue_by_pattern('Break here in struct')
        self.check_expression("a", "\"foo\"", use_summary=True)
        self.check_expression("self.b", "5", use_summary=False)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

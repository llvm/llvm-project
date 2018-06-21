# TestExprInProtocolExtension.py
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
Tests scoped variables with swift expressions
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftExprInProtocolExtension(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_protocol_extension(self):
        """Tests that swift expressions in protocol extension functions behave correctly"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def check_expression(self, expression, expected_result, use_summary=True):
        value = self.frame.EvaluateExpression(
            expression, lldb.eDynamicCanRunTarget)
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
        """Tests that swift expressions in protocol extension functions behave correctly"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.target = target
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        static_bkpt = target.BreakpointCreateBySourceRegex(
            'break here in static func', self.main_source_spec)
        self.assertTrue(static_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        method_bkpt = target.BreakpointCreateBySourceRegex(
            'break here in method', self.main_source_spec)
        self.assertTrue(method_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.process = process

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, method_bkpt)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        # Check that we can evaluate expressions correctly in the struct
        # method.
        self.check_expression("self.x", "10", False)
        self.check_expression("self.y", '"Hello world"', True)
        self.check_expression("local_var", "111", False)

        # And check that we got the type of self right:
        self_var = self.frame.EvaluateExpression(
            "self", lldb.eDynamicCanRunTarget)
        self_type_name = self_var.GetTypeName()
        print("Self type name is: ", self_type_name)

        # Not checking yet since we don't get this right.

        # Now continue to the static method and check things there:
        self.continue_to_bkpt(process, static_bkpt)

        self.check_expression("self.cvar", "333", False)
        self.check_expression("local_var", "222", False)

        # This continues to the class version:
        self.continue_to_bkpt(process, method_bkpt)
        # Check that we can evaluate expressions correctly in the struct
        # method.
        self.check_expression("self.x", "10", False)
        self.check_expression("self.y", '"Hello world"', True)
        self.check_expression("local_var", "111", False)

        # And check that we got the type of self right:
        self_var = self.frame.EvaluateExpression(
            "self", lldb.eDynamicCanRunTarget)
        self_type_name = self_var.GetTypeName()
        print("Self type name is: ", self_type_name)

        # Not checking yet since we don't get this right.

        # Now continue to the static method and check things there:
        self.continue_to_bkpt(process, static_bkpt)

        self.check_expression("self.cvar", "333", False)
        self.check_expression("local_var", "222", False)

        # This continues to the enum version:
        self.continue_to_bkpt(process, method_bkpt)
        # Check that we can evaluate expressions correctly in the struct
        # method.
        self.check_expression("self.x", "10", False)
        self.check_expression("self.y", '"Hello world"', True)
        self.check_expression("local_var", "111", False)

        # And check that we got the type of self right:
        self_var = self.frame.EvaluateExpression(
            "self", lldb.eDynamicCanRunTarget)
        self_type_name = self_var.GetTypeName()
        print("Self type name is: ", self_type_name)

        # Not checking yet since we don't get this right.

        # Now continue to the static method and check things there:
        self.continue_to_bkpt(process, static_bkpt)

        self.check_expression("self.cvar", "333", False)
        self.check_expression("local_var", "222", False)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

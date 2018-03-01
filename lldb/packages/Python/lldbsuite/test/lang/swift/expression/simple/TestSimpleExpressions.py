# TestSimpleExpressions.py
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
Tests simple swift expressions
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSimpleSwiftExpressions(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.add_test_categories(["swiftpr"])
    def test_simple_swift_expressions(self):
        """Tests that we can run simple Swift expressions correctly"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def check_expression(self, expression, expected_result, use_summary=True):
        value = self.frame.EvaluateExpression(expression)
        self.assertTrue(value.IsValid(), expression + "returned a valid value")
        if self.TraceOn():
            print value.GetSummary()
            print value.GetValue()
        if use_summary:
            answer = value.GetSummary()
        else:
            answer = value.GetValue()
        report_str = "%s expected: %s got: %s" % (
            expression, expected_result, answer)
        self.assertTrue(answer == expected_result, report_str)

    def do_test(self):
        """Test simple swift expressions"""
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

        # Test that parse errors give a correct result:
        value_obj = self.frame.EvaluateExpression(
            "iff is_five === 5 { return is_five")
        error = value_obj.GetError()

        # Test simple math with constants

        self.check_expression("5 + 6", "11", use_summary=False)
        self.check_expression("is_five + is_six", "11", use_summary=False)
        self.check_expression(
            "if (1 == 1) { return is_five + is_six }",
            "11",
            use_summary=False)

        # Test boolean operations with simple variables:
        # Bool's are currently enums, so their value is actually in the value.
        self.check_expression("is_eleven == is_five + is_six", "true")

        # Try a slightly more complex container for our expression:
        self.check_expression(
            "if is_five == 5 { return is_five + is_six } else { return is_five }",
            "11",
            use_summary=False)

        # Make sure we get an error if we don't give homogenous return types:
        bool_or_int = self.frame.EvaluateExpression(
            "if is_five == 5 { return is_five + is_six } else { return false }")
        self.assertTrue(
            bool_or_int.IsValid(),
            "if is_five == 5 { return is_five + is_six } else { return false } is invalid")

        # Make sure we get the correct branch of a complex result expression:
        self.check_expression(
            "if is_five == 6 {return is_five} else if is_six == 5 {return is_six} ; is_eleven",
            "11",
            use_summary=False)

        # Make sure we can access globals:
        # Commented out till we resolve <rdar://problem/15695494> Accessing global variables causes LLVM ERROR and exit...
        # self.check_expression ("my_global", "30")

        # Non-simple names:
        self.check_expression(
            u"\u20ac_varname".encode("utf-8"),
            "5",
            use_summary=False)

        # See if we can do the same manipulations with tuples:
        # Commented out due to: <rdar://problem/15476525> Expressions with
        # tuple elements assert
        self.check_expression("a_tuple.0 + a_tuple.1", "11", use_summary=False)

        # See if we can do some manipulations with dicts:
        self.check_expression(
            'str_int_dict["five"]! + str_int_dict["six"]!',
            "11",
            use_summary=False)
        self.check_expression(
            'int_str_dict[Int(is_five + is_six)]!',
            '"eleven"')
        # Commented out, touching the dict twice causes it to die, probably the same problem
        # as <rdar://problem/15306399>
        self.check_expression(
            'str_int_dict["five"] = 6; str_int_dict["five"]! + str_int_dict["six"]!',
            "12",
            use_summary=False)

        # See if we can use a switch statement in an expression:
        self.check_expression(
            "switch is_five { case 0..<6: return 1; case 7..<11: return 2; case _: return 4; }; 3;",
            "1",
            use_summary=False)

        # These ones are int-convertible and Equatable so we can do some things
        # with them anyway:
        self.check_expression("enum_eleven", "Eleven", False)
        self.check_expression("enum_eleven == SomeValues.Eleven", "true")
        self.check_expression(
            "SomeValues.Five.toInt() + SomeValues.Six.toInt()",
            "11",
            use_summary=False)
        self.check_expression(
            "enum_eleven = .Five; return enum_eleven == .Five", "true")

        # Test expressions with a simple object:
        self.check_expression("a_obj.x", "6", use_summary=False)
        # Should not have to make a second object here. This is another
        # side-effect of <rdar://problem/15306399>
        self.check_expression("a_nother_obj.y", "6.5", use_summary=False)

        # Test expressions with a struct:
        self.check_expression("b_struct.b_int", "5", use_summary=False)

        # Test expression with Chars and strings:

        # Commented out due to <rdar://problem/16856379>
        #self.check_expression ("a_char", "U+0061 U+0000 u'a'")

        # Interpolated strings and string addition:
        self.check_expression(
            '"Five: \(is_five) " + "Six: \(is_six)"',
            '"Five: 5 Six: 6"')

        # Next let's try some simple array accesses:
        self.check_expression("an_int_array[0]", "5", use_summary=False)

        # Test expression with read-only variables:
        self.check_expression("b_struct.b_read_only == 5", "true")
        failed_value = self.frame.EvaluateExpression(
            "b_struct.b_read_only = 34")
        self.assertTrue(failed_value.IsValid(),
                        "Get something back from the evaluation.")
        self.assertTrue(failed_value.GetError().Success()
                        == False, "But it is an error.")

        # Check a simple value in a struct:
        self.check_expression("b_struct_2.b_int", "20", use_summary=False)

        # Make sure this works for properties in extensions as well:
        self.check_expression("b_struct_2.b_float", "20.5", use_summary=False)

        # Here are a few tests of making variables in expressions:
        self.check_expression(
            "var enum_six : SomeValues = SomeValues.Six; return enum_six == .Six", "true")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

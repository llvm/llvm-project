# TestSwiftExpressionsInClassFunctions.py
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
Test expressions in the context of class functions
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftExpressionsInClassFunctions(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.swiftTest
    def test_expressions_in_class_functions(self):
        """Test expressions in class func contexts"""
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
        self.assertTrue(answer == expected_result, report_str)

    def do_test(self):
        """Test expressions in class func contexts"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoints = [None]

        # Set the breakpoints
        for i in range(1, 8):
            breakpoints.append(
                target.BreakpointCreateBySourceRegex(
                    "breakpoint " + str(i),
                    self.main_source_spec))
            self.assertTrue(
                breakpoints[i].GetNumLocations() > 0,
                "Didn't get valid breakpoint for %s" %
                (str(i)))

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Check each context
        for i in range(1, 8):
            # Frame #0 should be at our breakpoint.
            threads = lldbutil.get_threads_stopped_at_breakpoint(
                process, breakpoints[i])

            self.assertTrue(len(threads) == 1)
            self.thread = threads[0]
            self.frame = self.thread.frames[0]
            self.assertTrue(self.frame, "Frame 0 is valid.")

            self.check_expression("i", str(i), False)

            self.runCmd("continue")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

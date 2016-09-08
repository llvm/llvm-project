# TestInOutVariables.py
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
Test that @inout variables display reasonably
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestInOutVariables(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_in_out_variables(self):
        """Test that @inout variables display reasonably"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def check_class_internal(
            self,
            ivar_value,
            ovar_value,
            use_expression,
            is_wrapped):
        if use_expression:
            # At present the expression parser strips off the @lvalue bit:
            message_end = "from EvaluateExpression"
            x_actual = self.frame.EvaluateExpression(
                "x", lldb.eDynamicCanRunTarget)
            self.assertTrue(
                x_actual.GetError().Success(),
                "Expression evaluation failed: %s" %
                (x_actual.GetError().GetCString()))
        else:
            message_end = "from FindVariable"
            x = self.frame.FindVariable("x").GetDynamicValue(
                lldb.eDynamicCanRunTarget)
            self.assertTrue(x.IsValid(), "did not find x %s" % (message_end))
            if is_wrapped:
                self.assertTrue(
                    x.GetNumChildren() == 1,
                    "x has too many children %s" %
                    (message_end))
                x_actual = x.GetChildAtIndex(0).GetDynamicValue(
                    lldb.eDynamicCanRunTarget)
                self.assertTrue(
                    x_actual.IsValid(),
                    "did not find the child of x %s" %
                    (message_end))
            else:
                x_actual = x

        ivar = x_actual.GetChildAtIndex(0).GetChildAtIndex(0)
        ovar = x_actual.GetChildAtIndex(1)
        self.assertTrue(
            ivar.GetName() == "ivar", "Name: %s is not ivar %s" %
            (ivar.GetName(), message_end))
        self.assertTrue(
            ovar.GetName() == "ovar",
            "ovar is not ovar %s" %
            (message_end))
        self.assertTrue(
            ivar.GetValue() == ivar_value,
            "ivar wrong %s" %
            (message_end))
        self.assertTrue(
            ovar.GetValue() == ovar_value,
            "ovar wrong %s" %
            (message_end))

    def check_class(self, ivar_value, ovar_value, is_wrapped=True):
        self.check_class_internal(ivar_value, ovar_value, True, is_wrapped)
        self.check_class_internal(ivar_value, ovar_value, False, is_wrapped)

    def check_struct_internal(self, ivar_value, use_expression):
        if use_expression:
            x = self.frame.EvaluateExpression("x", lldb.eDynamicCanRunTarget)
            message_end = "from EvaluateExpression"
        else:
            x = self.frame.FindVariable("x").GetDynamicValue(
                lldb.eDynamicCanRunTarget)
            message_end = "from FindVariable"

        self.assertTrue(x.IsValid(), "did not find x %s" % (message_end))
        self.assertTrue(
            x.GetNumChildren() == 1,
            "x has too many children %s" %
            (message_end))
        ivar = x.GetChildAtIndex(0)
        if not use_expression:
            ivar = ivar.GetChildAtIndex(0)
        self.assertTrue(
            ivar.GetName() == "ivar",
            "ivar is not ivar %s" %
            (message_end))
        self.assertTrue(
            ivar.GetValue() == ivar_value,
            "ivar wrong %s" %
            (message_end))

    def check_struct(self, ivar_value):
        self.check_struct_internal(ivar_value, True)
        self.check_struct_internal(ivar_value, False)

    def check_next_stop(self, breakpoint):
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            self.process, breakpoint)
        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

    def do_test(self):
        """Test that @inout variables display reasonably"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        class_bkpt = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here for Class access', self.main_source_spec)
        self.assertTrue(class_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        struct_bkpt = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here for Struct access', self.main_source_spec)
        self.assertTrue(struct_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        outer_bkpt = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here for String access', self.main_source_spec)
        self.assertTrue(outer_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        self.process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(self.process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        self.check_next_stop(class_bkpt)
        self.check_class("4321", "112233")

        self.process.Continue()
        self.check_next_stop(struct_bkpt)
        self.check_struct("4567")

        self.process.Continue()
        self.check_next_stop(class_bkpt)
        self.check_class("4322", "112233")

        # Now on the way out let's modify the class and make sure it gets
        # modified both here and outside the functions:
        var = self.frame.EvaluateExpression(
            "x = Other(in1: 556678, in2: 667788)",
            lldb.eDynamicCanRunTarget)
        self.assertTrue(var.GetError().Success())

        #self.check_class("556677", "667788")

        self.process.Continue()
        self.check_next_stop(outer_bkpt)

        svar = self.frame.FindVariable("x").GetChildAtIndex(0)
        self.assertTrue(svar.GetSummary() == '"Keep going, nothing to see"')

        self.process.Continue()
        self.check_next_stop(class_bkpt)
        self.check_class("556679", "667788")

        self.process.Continue()
        self.check_next_stop(struct_bkpt)
        self.check_struct("4568")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

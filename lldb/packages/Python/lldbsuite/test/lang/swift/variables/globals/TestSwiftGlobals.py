# TestSwiftGlobals.py
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
Check that we can examine module globals in the expression parser.
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftGlobals(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_swift_globals(self):
        """Check that we can examine module globals in the expression parser"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Check that we can examine module globals in the expression parser"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        outer_bkpt = target.BreakpointCreateBySourceRegex(
            'Set top_level breakpoint here', self.main_source_spec)
        self.assertTrue(outer_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        function_bkpt = target.BreakpointCreateBySourceRegex(
            'Set function breakpoint here', self.main_source_spec)
        self.assertTrue(function_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, outer_bkpt)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]

        # All the variables should be uninitialized at this point.  Maybe sure
        # they look that way:
        frame = self.thread.frames[0]
        options = lldb.SBExpressionOptions()
        options.SetFetchDynamicValue(lldb.eDynamicCanRunTarget)

        error = lldb.SBError()

        # Examine the variables before initialization:

        g_counter = frame.EvaluateExpression("g_counter", options)
        self.assertTrue(
            g_counter.IsValid(),
            "g_counter returned a valid value object.")
        value = g_counter.GetValueAsSigned(error)
        self.assertTrue(error.Success(), "Got a value for g_counter")
        self.assertTrue(
            value == 0,
            "g_counter value is the uninitialized one.")

        foo_var = frame.EvaluateExpression("my_foo", options)
        self.assertTrue(
            foo_var.IsValid(),
            "foo_var returned a valid value object.")
        value = foo_var.GetValueAsUnsigned(error)
        self.assertTrue(error.Success(), "foo_var has a value.")
        self.assertTrue(value == 0, "foo_var is null before initialization.")

        my_large_dude = frame.EvaluateExpression("my_large_dude", options)
        self.assertTrue(my_large_dude.IsValid(),
                        "my_large_dude returned a valid value object.")
        value = my_large_dude.GetValue()
        self.assertTrue(error.Success(), "Got a value for my_large_dude")
        self.assertTrue(
            value is None,
            "my_large_dude value is the uninitialized one.")

        # Now proceed to the breakpoint in our main function, make sure we can
        # still read these variables and they now have the right values.
        threads = lldbutil.continue_to_breakpoint(process, function_bkpt)
        self.assertTrue(len(threads) == 1)

        self.thread = threads[0]

        # Examine the variables before initialization:

        g_counter = frame.EvaluateExpression("g_counter", options)
        self.assertTrue(
            g_counter.IsValid(),
            "g_counter returned a valid value object.")
        value = g_counter.GetValueAsSigned(error)
        self.assertTrue(error.Success(), "Got a value for g_counter")
        self.assertTrue(value == 2, "g_counter value should be 2.")

        foo_var = frame.EvaluateExpression("my_foo", options)
        self.assertTrue(
            foo_var.IsValid(),
            "foo_var returned a valid value object.")
        foo_var_x = foo_var.GetChildMemberWithName("x")
        self.assertTrue(foo_var_x.IsValid(), "Got value object for foo_var.x")
        value = foo_var_x.GetValueAsUnsigned(error)
        self.assertTrue(error.Success(), "foo_var.x has a value.")
        self.assertTrue(value == 1, "foo_var is null before initialization.")

        my_large_dude = frame.EvaluateExpression("my_large_dude", options)
        self.assertTrue(my_large_dude.IsValid(),
                        "my_large_dude returned a valid value object.")
        my_large_dude_y = my_large_dude.GetChildMemberWithName("y")
        self.assertTrue(
            my_large_dude_y.IsValid(),
            "Got value object for my_large_dude.y")
        value = my_large_dude_y.GetValueAsUnsigned(error)
        self.assertTrue(error.Success(), "Got a value for my_large_dude.y")
        self.assertTrue(
            value == 20,
            "my_large_dude value is the uninitialized one.")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

# TestEqualityOperators.py
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
Test that we resolve various shadowed equality operators properly
"""
import commands
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import no_match
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import unittest2


def execute_command(command):
    (exit_status, output) = commands.getstatusoutput(command)
    return exit_status


class TestUnitTests(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.skipIf(debug_info=no_match(["dsym"]), bugnumber="This test is building a dSYM")
    @decorators.expectedFailureAll(
        oslist=["linux"],
        bugnumber="rdar://28180489")
    def test_equality_operators_fileprivate(self):
        """Test that we resolve expression operators correctly"""
        self.build()
        self.do_test("Fooey.CompareEm1", "true", 1)

    @decorators.swiftTest
    @decorators.skipIf(debug_info=no_match(["dsym"]), bugnumber="This test is building a dSYM")
    @decorators.expectedFailureAll(
        oslist=["linux"],
        bugnumber="rdar://28180489")
    def test_equality_operators_private(self):
        """Test that we resolve expression operators correctly"""
        self.build()
        self.do_test("Fooey.CompareEm2", "false", 2)

    @decorators.swiftTest
    @decorators.skipIf(debug_info=no_match(["dsym"]), bugnumber="This test is building a dSYM")
    @decorators.expectedFailureAll(
        oslist=["linux"],
        bugnumber="rdar://28180489")
    @decorators.expectedFailureAll(bugnumber="rdar://38483465")
    def test_equality_operators_other_module(self):
        """Test that we resolve expression operators correctly"""
        self.build()
        self.do_test("Fooey.CompareEm3", "false", 3)

    def setUp(self):
        TestBase.setUp(self)

    def do_test(self, bkpt_name, compare_value, counter_value):
        """Test that we resolve expression operators correctly"""
        exe_name = "three"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        bkpt = target.BreakpointCreateByName(bkpt_name)
        self.assertTrue(bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        env_arr = self.registerSharedLibrariesWithTarget(target, ["fooey"])

        process = target.LaunchSimple(None, env_arr, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        threads = lldbutil.get_threads_stopped_at_breakpoint(process, bkpt)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        options = lldb.SBExpressionOptions()

        value = self.frame.EvaluateExpression("lhs == rhs", options)
        self.assertTrue(
            value.GetError().Success(),
            "Expression in %s was successful" %
            (bkpt_name))
        summary = value.GetSummary()
        self.assertTrue(
            summary == compare_value,
            "Expression in CompareEm has wrong value: %s (expected %s)." %
            (summary,
             compare_value))

        # And make sure we got did increment the counter by the right value.
        value = self.frame.EvaluateExpression("Fooey.GetCounter()", options)
        self.assertTrue(value.GetError().Success(), "GetCounter failed with error %s"%(value.GetError().GetCString()))

        counter = value.GetValueAsUnsigned()
        self.assertTrue(
            counter == counter_value, "Counter value is wrong: %d (expected %d)" %
            (counter, counter_value))

        # Make sure the presence of these type specific == operators doesn't interfere
        # with finding other unrelated == operators.
        value = self.frame.EvaluateExpression("1 == 2", options)
        self.assertTrue(
            value.GetError().Success(),
            "1 == 2 expression couldn't run")
        self.assertTrue(
            value.GetSummary() == "false",
            "1 == 2 didn't return false.")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

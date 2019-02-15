"""
Test expression operations in class constrained protocols
"""

from __future__ import print_function


import os
import time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class TestClassConstrainedProtocol(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_extension_weak_self (self):
        """Test that we can reconstruct weak self captured in a class constrained protocol."""
        self.build()
        self.do_self_test("Break here for weak self")

    @decorators.swiftTest
    @expectedFailureAll(oslist=["linux"], bugnumber="rdar://31822722")
    def test_extension_self (self):
        """Test that we can reconstruct self in method of a class constrained protocol."""
        self.build()
        self.do_self_test("Break here in class protocol")

    @decorators.swiftTest
    def test_method_weak_self (self):
        """Test that we can reconstruct weak self capture in method of a class conforming to a class constrained protocol."""
        self.build()
        self.do_self_test("Break here for method weak self")

    @decorators.swiftTest
    def test_method_self (self):
        """Test that we can reconstruct self in method of a class conforming to a class constrained protocol."""
        self.build()
        self.do_self_test("Break here in method")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def get_to_bkpt(self, bkpt_pattern):
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint in main.c at the source matching
        # bkpt_pattern
        breakpoint = target.BreakpointCreateBySourceRegex(
            bkpt_pattern, lldb.SBFileSpec("main.swift"))
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() >= 1,
                        VALID_BREAKPOINT)

        error = lldb.SBError()
        # This is the launch info.  If you want to launch with arguments or
        # environment variables, add them using SetArguments or
        # SetEnvironmentEntries

        launch_info = lldb.SBLaunchInfo(None)
        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        # Did we hit our breakpoint?
        from lldbsuite.test.lldbutil import get_threads_stopped_at_breakpoint
        threads = get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertTrue(
            len(threads) == 1,
            "There should be a thread stopped at our breakpoint")

        # The hit count for the breakpoint should be 1.
        self.assertTrue(breakpoint.GetHitCount() == 1)

        self.frame = threads[0].GetFrameAtIndex(0)

    def check_self(self, bkpt_pattern):
        opts = lldb.SBExpressionOptions()
        result = self.frame.EvaluateExpression("self", opts)
        error = result.GetError()
        self.assertTrue(error.Success(), "'self' expression failed at '%s': %s"%(bkpt_pattern, error.GetCString()))
        f_ivar = result.GetChildMemberWithName("f")
        self.assertTrue(f_ivar.IsValid(), "Could not find 'f' in self at '%s'"%(bkpt_pattern))
        self.assertTrue(f_ivar.GetValueAsSigned() == 12345, "Wrong value for f: %d"%(f_ivar.GetValueAsSigned()))

    def do_self_test(self, bkpt_pattern):
        self.get_to_bkpt(bkpt_pattern)
        self.check_self(bkpt_pattern)

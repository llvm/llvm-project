"""
Test that breakpoints set on a bad address say they are bad.
"""

from __future__ import print_function


import os
import time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class BadAddressBreakpointTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def test_bad_address_breakpoints(self):
        """Test that breakpoints set on a bad address say they are bad."""
        self.build()
        self.address_breakpoints()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def address_breakpoints(self):
        """Test that breakpoints set on a bad address say they are bad."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateBySourceRegex(
            "Set a breakpoint here", lldb.SBFileSpec("main.c"))
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Get the breakpoint location from breakpoint after we verified that,
        # indeed, it has one location.
        location = breakpoint.GetLocationAtIndex(0)
        self.assertTrue(location and
                        location.IsEnabled(),
                        VALID_BREAKPOINT_LOCATION)

        launch_info = lldb.SBLaunchInfo(None)

        error = lldb.SBError()

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

        # Now see if we can read from 0.  If I can't do that, I don't have a good way to know
        # what an illegal address is...
        error.Clear()

        ptr = process.ReadPointerFromMemory(0x0, error)

        if not error.Success():
            bkpt = target.BreakpointCreateByAddress(0x0)
            for bp_loc in bkpt:
                self.assertTrue(bp_loc.IsResolved() == False)
        else:
            self.fail(
                "Could not find an illegal address at which to set a bad breakpoint.")

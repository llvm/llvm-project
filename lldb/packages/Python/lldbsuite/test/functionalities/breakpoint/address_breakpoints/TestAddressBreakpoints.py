"""
Test address breakpoints set with shared library of SBAddress work correctly.
"""

from __future__ import print_function


import os
import time
import re
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class AddressBreakpointTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def test_address_breakpoints(self):
        """Test address breakpoints set with shared library of SBAddress work correctly."""
        self.build()
        self.address_breakpoints()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def address_breakpoints(self):
        """Test address breakpoints set with shared library of SBAddress work correctly."""
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateBySourceRegex(
            "Set a breakpoint here", lldb.SBFileSpec("main.c"))
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() >= 1,
                        VALID_BREAKPOINT)

        # Get the breakpoint location from breakpoint after we verified that,
        # indeed, it has one location.
        location = breakpoint.GetLocationAtIndex(0)
        self.assertTrue(location and
                        location.IsEnabled(),
                        VALID_BREAKPOINT_LOCATION)

        # Next get the address from the location, and create an address breakpoint using
        # that address:

        address = location.GetAddress()
        target.BreakpointDelete(breakpoint.GetID())

        breakpoint = target.BreakpointCreateBySBAddress(address)

        # Disable ASLR.  This will allow us to actually test (on platforms that support this flag)
        # that the breakpoint was able to track the module.

        launch_info = lldb.SBLaunchInfo(None)
        flags = launch_info.GetLaunchFlags()
        flags &= ~lldb.eLaunchFlagDisableASLR
        launch_info.SetLaunchFlags(flags)

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

        process.Kill()

        # Now re-launch and see that we hit the breakpoint again:
        launch_info.Clear()
        launch_info.SetLaunchFlags(flags)

        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        thread = get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertTrue(
            len(threads) == 1,
            "There should be a thread stopped at our breakpoint")

        # The hit count for the breakpoint should now be 2.
        self.assertTrue(breakpoint.GetHitCount() == 2)

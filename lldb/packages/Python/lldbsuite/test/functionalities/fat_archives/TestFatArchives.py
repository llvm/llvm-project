"""
Test some lldb command abbreviations.
"""
from __future__ import print_function


import lldb
import os
import time
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class FatArchiveTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    def test_breakpoint_resolution_dwarf(self):
        if self.getArchitecture() == 'x86_64':
            self.build()
            self.main()
        else:
            self.skipTest(
                "This test requires x86_64 as the architecture for the inferior")

    def main(self):
        '''This test compiles a quick example by making a fat file (universal) full of
        skinny .o files and makes sure we can use them to resolve breakpoints when doing
        DWARF in .o file debugging. The only thing this test needs to do is to compile and
        set a breakpoint in the target and verify any breakpoint locations have valid debug
        info for the function, and source file and line.'''
        exe = self.getBuildArtifact("a.out")

        # Create the target
        target = self.dbg.CreateTarget(exe)

        # Create a breakpoint by name
        breakpoint = target.BreakpointCreateByName('foo', exe)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Make sure the breakpoint resolves to a function, file and line
        for bp_loc in breakpoint:
            # Get a section offset address (lldb.SBAddress) from the breakpoint
            # location
            bp_loc_addr = bp_loc.GetAddress()
            line_entry = bp_loc_addr.GetLineEntry()
            function = bp_loc_addr.GetFunction()
            self.assertTrue(
                function.IsValid(),
                "Verify breakpoint in fat BSD archive has valid function debug info")
            self.assertTrue(
                line_entry.GetFileSpec(),
                "Verify breakpoint in fat BSD archive has source file information")
            self.assertTrue(
                line_entry.GetLine() != 0,
                "Verify breakpoint in fat BSD archive has source line information")

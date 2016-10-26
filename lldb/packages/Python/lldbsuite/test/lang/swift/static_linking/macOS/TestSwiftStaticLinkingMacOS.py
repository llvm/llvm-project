# TestSwiftStaticLinkingMacOS.py
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
Test that macOS can statically link two separately-compiled Swift modules
with one Objective-C module, link them through the clang driver, and still
access debug info for each of the Swift modules.
"""
from __future__ import print_function


# System imports
import os
import commands

# Third-party imports

# LLDB imports
import lldb
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import decorators, lldbtest, lldbutil


class SwiftStaticLinkingMacOSTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def expect_self_var_available_at_breakpoint(
            self, process, breakpoint, module_name):
        # Frame #0 should be at the given breakpoint
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertEquals(1, len(threads))
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        patterns = [
            # Ensure we report a self with an address.
            r"self\s*=\s*0x[0-9a-fA-F]+",
            # Ensure we think it is an NSObject.
            r"ObjectiveC.NSObject"]
        substrs = [
            "(%s.%s)" % (module_name, module_name)
        ]
        self.expect("frame variable self", patterns=patterns,
                    substrs=substrs)

    @decorators.skipUnlessDarwin
    def test_variables_print_from_both_swift_modules(self):
        """Test that variables from two modules can be accessed."""
        self.build()

        # I could not find a reasonable way to say "skipUnless(archs=[])".
        # That would probably be worth adding.
        if self.getArchitecture() != 'x86_64':
            self.skipTest("This test requires x86_64 as the architecture "
                          "for the inferior")

        exe_name = "main"
        src_a = lldb.SBFileSpec("A.swift")
        line_a = 5
        src_b = lldb.SBFileSpec("B.swift")
        line_b = 5
        exe = os.path.join(os.getcwd(), exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        # Set the breakpoints
        # breakpoint_a = target.BreakpointCreateBySourceRegex(
        #     'Set breakpoint here', src_a)
        breakpoint_a = target.BreakpointCreateByLocation(
            src_a, line_a)
        self.assertTrue(breakpoint_a.GetNumLocations() > 0,
                        lldbtest.VALID_BREAKPOINT)

        # breakpoint_b = target.BreakpointCreateBySourceRegex(
        #     'Set breakpoint here', src_b)
        breakpoint_b = target.BreakpointCreateByLocation(
            src_b, line_b)
        self.assertTrue(breakpoint_b.GetNumLocations() > 0,
                        lldbtest.VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        envp = ['DYLD_FRAMEWORK_PATH=.']
        process = target.LaunchSimple(None, envp, os.getcwd())

        self.assertTrue(process, lldbtest.PROCESS_IS_VALID)

        # We should be at breakpoint in module A.
        self.expect_self_var_available_at_breakpoint(
            process, breakpoint_a, "A")

        # Jump to the next breakpoint
        process.Continue()

        # We should be at breakpoint in module B.
        self.expect_self_var_available_at_breakpoint(
            process, breakpoint_b, "B")

        return

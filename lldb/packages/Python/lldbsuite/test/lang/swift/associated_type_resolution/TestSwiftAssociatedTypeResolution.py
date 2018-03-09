# TestSwiftAssociatedTypeResolution.py
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
Test that associated-typed objects get resolved to their proper location in memory
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftArchetypeResolution(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_swift_associated_type_resolution(self):
        """Test that associated-typed objects get resolved to their proper location in memory"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Test that archetype-typed objects get resolved to their proper location in memory"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set a breakpoint here', self.main_source_spec)
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

        var = self.frame.FindVariable("things")
        var.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        var.SetPreferSyntheticValue(True)
        self.assertTrue(var.GetError().Success(), "Failed to get things: %s"%(var.GetError().GetCString()))
        self.assertEqual(var.GetNumChildren(), 4, "Got the right number of children")
        type_name = var.GetTypeName()
        self.assertEqual(type_name, "Swift.Array<Swift.Int>", "Wrong typename: %s."%(type_name))
        for i in range(0,4):
            child = var.GetChildAtIndex(i)
            self.assertTrue(child.GetError().Success(), "Failed to get things[%d]: %s"%(i, var.GetError().GetCString()))
            value = child.GetValueAsUnsigned()
            self.assertEqual(value, i, "Wrong value: %d not %d."%(value, i))

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

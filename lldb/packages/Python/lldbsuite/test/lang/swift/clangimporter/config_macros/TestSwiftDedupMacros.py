# TestSwiftDedupMacros.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2018 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------

import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2

class TestSwiftDedupMacros(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    # NOTE: rdar://44201206 - This test may sporadically segfault. It's likely
    # that the underlying memory corruption issue has been addressed, but due
    # to the difficulty of reproducing the crash, we are not sure. If a crash
    # is observed, try to collect a crashlog before disabling this test.
    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    @decorators.add_test_categories(["swiftpr"])
    def testSwiftDebugMacros(self):
        """This tests that configuration macros get uniqued when building the
        scratch ast context. Note that "-D MACRO" options with a space
        are currently only combined to "-DMACRO" when they appear
        outside of the main binary.

        """
        self.build()
            
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints.
        foo_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('dylib.swift'))

        process = target.LaunchSimple(None, None, os.getcwd())

        # Turn on logging.
        command_result = lldb.SBCommandReturnObject()
        interpreter = self.dbg.GetCommandInterpreter()
        log = self.getBuildArtifact("types.log")
        interpreter.HandleCommand("log enable lldb types -f "+log, command_result)
        
        self.expect("p foo", DATA_TYPES_DISPLAYED_CORRECTLY, substrs=["42"])
        debug = 0
        space = 0
        ndebug = 0
        space_with_space = 0
        logfile = open(log, "r")
        for line in logfile:
            if "-DDEBUG=1" in line:
                debug += 1
            if "-DSPACE" in line:
                space += 1
            if " SPACE" in line:
                space_with_space += 1
            if "-UNDEBUG" in line:
                ndebug += 1
        self.assertTrue(debug == 1)
        self.assertTrue(space == 1)
        self.assertTrue(space_with_space == 0)
        self.assertTrue(ndebug == 1)
        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

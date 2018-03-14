# TestSwiftHeadermaps.py
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


class TestSwiftHeadermaps(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    def run_to_breakpoint(self, use_hmaps):
        # We need to use separate module caches here.
        if use_hmaps:
            mod_cache = self.getBuildArtifact("module-cache-true")
            self.runCmd("settings set symbols.swift-use-headermaps true")
        else:
            mod_cache = self.getBuildArtifact("module-cache-false")
            self.runCmd("settings set symbols.swift-use-headermaps false")

        self.runCmd('settings set symbols.clang-modules-cache-path "%s"' % mod_cache)

        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

        # Create the target
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = threads[0].frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    @decorators.add_test_categories(["swiftpr"])
    def test_without_hmap(self):
        self.build()
        self.run_to_breakpoint(False)

        # This expression should only work if headermaps lookup is enabled.
        value = self.frame.EvaluateExpression("foo")
        self.assertFalse(value.GetError().Success())

    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    @decorators.add_test_categories(["swiftpr"])
    def test_with_hmap(self):
        self.build()
        self.run_to_breakpoint(True)

        # This expression should only work if headermaps lookup is enabled.
        self.expect("expr foo", substrs=['42'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

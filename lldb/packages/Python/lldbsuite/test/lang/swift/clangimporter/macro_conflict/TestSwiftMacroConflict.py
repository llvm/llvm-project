# TestSwiftMacroConflict.py
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
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2
import shutil

class TestSwiftMacroConflict(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    @skipUnlessDarwin
    @swiftTest
    def test(self):
        # To ensure we hit the rebuild problem remove the cache to avoid caching.
        mod_cache = self.getBuildArtifact("my-clang-modules-cache")
        if os.path.isdir(mod_cache):
          shutil.rmtree(mod_cache)

        self.runCmd('settings set symbols.clang-modules-cache-path "%s"'
                    % mod_cache)
        self.build()
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        bar_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('Bar.swift'))

        process = target.LaunchSimple(None, None, os.getcwd())

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, bar_breakpoint)
        frame = threads[0].GetFrameAtIndex(0)
        bar_value = frame.EvaluateExpression("bar")
        self.expect("fr var bar", "correct bar", substrs=["23"])

        foo_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('Foo.swift'))
        process.Continue()

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, foo_breakpoint)
        frame = threads[0].GetFrameAtIndex(0)
        foo_value = frame.EvaluateExpression("foo")
        # One is expected to fail because we use the same -D define as above.
        self.assertTrue(foo_value.GetError().Success() ^
                        bar_value.GetError().Success())

        self.assertTrue(os.path.isdir(mod_cache), "module cache exists")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

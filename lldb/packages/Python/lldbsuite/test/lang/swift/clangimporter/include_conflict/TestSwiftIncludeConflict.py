# TestSwiftHeadermapConflict.py
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
import shutil

class TestSwiftIncludeConflict(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    @decorators.add_test_categories(["swiftpr"])
    def test(self):
        # To ensure we hit the rebuild problem remove the cache to avoid caching.
        mod_cache = self.getBuildArtifact("my-clang-modules-cache")
        if os.path.isdir(mod_cache):
          shutil.rmtree(mod_cache)

        self.runCmd("settings set symbols.use-swift-dwarfimporter false")
        self.runCmd('settings set symbols.clang-modules-cache-path "%s"'
                    % mod_cache)
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        b_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('dylib.swift'))

        # This is expected to succeed because ClangImporter was set up
        # with the flags from the main executable.
        self.expect("expr foo", "expected result", substrs=["42"])
        self.assertTrue(os.path.isdir(mod_cache), "module cache exists")

        process.Continue()
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, b_breakpoint)
        frame = threads[0].GetFrameAtIndex(0)
        value = frame.EvaluateExpression("foo")
        # This is expected to fail because ClangImporter is *still*
        # set up with the flags from the main executable.
        self.assertTrue((not value.GetError().Success()) or
                         not value.GetSummary())
        self.runCmd("settings set symbols.use-swift-dwarfimporter true")
        

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

# TestSwiftObjCMainConflictingDylibsFailingImport.py
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

class TestSwiftObjCMainConflictingDylibsFailingImport(TestBase):

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

        # This is failing because the Target-SwiftASTContext uses the
        # amalgamated target header search options from all dylibs.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, bar_breakpoint)

        # This works because the Module-SwiftASTContext uses the dylib flags.
        self.expect("fr var bar", "expected result", substrs=["42"])
        self.assertTrue(os.path.isdir(mod_cache), "module cache exists")
        # This initially fails with the shared scratch context and is
        # then retried with the per-dylib scratch context.
        self.expect("p bar", "expected result", substrs=["$R0", "42"])
        self.expect("p $R0", "expected result", substrs=["$R2", "42"])
        self.expect("p $R2", "expected result", substrs=["$R4", "42"])
        
        # This works by accident because the search paths are in the right order.
        foo_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('Foo.swift'))
        process.Continue()
        self.expect("fr var foo", "expected result", substrs=["23"])
        self.expect("p foo", "expected result", substrs=["23"])
        self.expect("p $R6", "expected result", substrs=["23"])
        self.expect("p $R8", "expected result", substrs=["23"])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

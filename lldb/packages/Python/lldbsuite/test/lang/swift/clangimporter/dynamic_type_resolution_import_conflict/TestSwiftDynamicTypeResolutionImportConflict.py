# TestSwiftDynamicTypeResolutionImportConflict.py
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

class TestSwiftDynamicTypeResolutionImportConflict(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """
        This testcase causes the scratch context to get destroyed by a
        conflict that is triggered via dynamic type resolution. The
        conflict is triggered by "frame variable" alone. The final
        "expr" command is just there to test that after "fr var" has
        destroyed the scratch context we can recover.

        """
        # To ensure we hit the rebuild problem remove the cache to avoid caching.
        mod_cache = self.getBuildArtifact("my-clang-modules-cache")
        if os.path.isdir(mod_cache):
          shutil.rmtree(mod_cache)

        self.runCmd('settings set symbols.clang-modules-cache-path "%s"'
                    % mod_cache)
        self.build()

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.registerSharedLibrariesWithTarget(target, ['Dylib', 'Conflict'])

        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.swift'))
        # Destroy the scratch context with a dynamic type lookup.
        self.expect("target var -d run-target -- foofoo",
                    substrs=['(Conflict.C) foofoo'])
        self.expect("target var -- foofoo",
                    substrs=['(Conflict.C) foofoo'])
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('Dylib.swift'))
        self.expect("bt", substrs=['Dylib.swift'])
        self.expect("fr v -d no-dynamic-values -- input",
                    substrs=['(Dylib.LibraryProtocol) input'])
        self.expect("fr v -d run-target -- input",
                    substrs=['(a.FromMainModule) input'])
        self.expect("expr -d run-target -- input",
                    substrs=['(a.FromMainModule) $R0'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

# TestSwiftRemoteASTImport.py
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

class TestSwiftRemoteASTImport(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    
    def setUp(self):
        TestBase.setUp(self)

    @decorators.skipUnlessDarwin
    @decorators.swiftTest
    @decorators.add_test_categories(["swiftpr"])
    def testSwiftRemoteASTImport(self):
        """This tests that RemoteAST querying the dynamic type of a variable
        doesn't import any modules into a module SwiftASTContext that
        weren't imported by that module in the source code.

        """
        self.build()
        # The Makefile doesn't build a .dSYM, so we need to help with
        # finding the .swiftmodules.
        os.chdir(self.getBuildDir())
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('Library.swift'))
        # FIXME: Reversing the order of these two commands does not work!
        self.expect("expr -d no-dynamic-values -- input",
                    substrs=['(LibraryProtocol) $R0'])
        self.expect("expr -d run-target -- input",
                    substrs=['(a.FromMainModule) $R2'])
        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

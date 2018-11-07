# TestSwiftModuleSearchPaths.py
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
Tests that we can import modules located using target.swift-module-search-paths
"""

import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestSwiftModuleSearchPaths(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)


    @decorators.swiftTest
    @decorators.add_test_categories(["swiftpr"])
    def test_swift_module_search_paths(self):
        """
        Tests that we can import modules located using
        target.swift-module-search-paths
        """
        
        # Build and run the dummy target
        self.build()
        
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.swift'))

        # Add the current working dir to the swift-module-search-paths
        self.runCmd("settings append target.swift-module-search-paths " +
                    self.getBuildDir())
        
        # import the module
        self.runCmd("e import Module")
        
        # Check that we know about the function declared in the module
        self.match(
            "e plusTen(10)", "error: Couldn't lookup symbols:", error=True)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

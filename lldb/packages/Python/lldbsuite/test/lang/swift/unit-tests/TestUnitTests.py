# TestUnitTests.py
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
Test that XCTest-based unit tests work
"""
import commands
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import unittest2


class TestUnitTests(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @swiftTest
    # The creation of the .xctest framework messes with the AST search path.
    @skipIf(debug_info=decorators.no_match("dsym"))
    def test_cross_module_extension(self):
        """Test that XCTest-based unit tests work"""
        self.build()
        self.do_test()

    def do_test(self):
        """Test that XCTest-based unit tests work"""
        lldbutil.run_to_source_breakpoint(self,
                                          "Set breakpoint here",
                                          lldb.SBFileSpec('xctest.c'),
                                          exe_name = "xctest")

        options = lldb.SBExpressionOptions()
        options.SetLanguage(lldb.eLanguageTypeSwift)
        self.expect("log enable lldb host -f ~/Desktop/t1.log")
        self.expect("expr -l Swift -- import test")
        self.expect("expr -l Swift -- doTest()",
                    substrs=['Int','$R0','=','3'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

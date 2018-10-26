# TestSwiftUnitTests.py
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
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftUnitTests(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @swiftTest
    # The creation of the .xctest framework messes with the AST search path.
    @skipIf(debug_info=no_match("dsym"))
    @add_test_categories(["swiftpr"])
    def test_cross_module_extension(self):
        """Test that XCTest-based unit tests work"""
        self.build()
        lldbutil.run_to_source_breakpoint(self,
                                          "Set breakpoint here",
                                          lldb.SBFileSpec('xctest.c'),
                                          exe_name = "xctest")
        self.expect("expr -l Swift -- import test")
        self.expect("expr -l Swift -- doTest()",
                    substrs=['Int','$R0','=','3'])

# TestSwiftMetadataSymbols.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See http://swift.org/LICENSE.txt for license information
# See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Test that swift metadata symbols are printed and classified correctly
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os


class TestSwiftMetadataSymbols(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_swift_metadata_symbols(self):
        """Test swift Class types"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)

    def do_test(self):
        """Tests that we can break and display simple types"""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        self.expect("target modules dump symtab -m a.out",
                    patterns=['Metadata.*_TMC1a3Foo'])

        self.expect("target modules dump symtab a.out",
                    patterns=['Metadata.*type metadata for'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

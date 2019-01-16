# TestSwiftBackwardsCompatibilitySimple.py
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
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftBackwardsCompatibilitySimple(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    @skipIf(compiler="swiftc", compiler_version=['<', '5.0'])
    @add_test_categories(["swiftpr", "swift-history"])
    def test_simple(self):
        version = self.getCompilerVersion(os.environ['SWIFTC'])
        if version < '5.0':
            self.skipTest('Swift compiler predates stable ABI')
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.swift'))
        # FIXME: Removing the next line breaks subsequent expressions
        #        when the swiftmodules can't be loaded.
        self.expect("fr v")
        self.expect("fr v number", substrs=['23'])
        self.expect("fr v array", substrs=['1', '2', '3'])
        self.expect("fr v string", substrs=['"hello"'])
        self.expect("fr v tuple", substrs=['42', '"abc"'])
        # FIXME: This doesn't work.
        #self.expect("fr v generic", substrs=['-1'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lldb.SBDebugger.Terminate)
    unittest2.main()

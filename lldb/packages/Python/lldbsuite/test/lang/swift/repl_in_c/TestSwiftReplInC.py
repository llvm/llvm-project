# TestSwiftReplInC.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2018 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftReplInC(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @add_test_categories(["swiftpr"])
    def test_repl_in_c(self):
        self.build()
        lldbutil.run_to_name_breakpoint(self, "main")
        self.expect("repl", error=True, substrs=["Swift standard library"])
        self.runCmd("kill")
        self.expect("repl", error=True, substrs=["running process"])

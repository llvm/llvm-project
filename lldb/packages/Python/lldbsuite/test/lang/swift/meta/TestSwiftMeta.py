# TestSwiftMeta.py
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
"""
Test the Swift test decorator itself.
"""
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbtest as lldbtest
import os

class TestSwiftMeta(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    def test_swiftDecorator(self):
        self.assertTrue(self.getDebugInfo() <> "gmodules")

    @decorators.swiftTest
    def test_swiftBuild(self):
        self.build()
        exe = self.getBuildArtifact()
        dsym = exe+'.dSYM'
        self.assertTrue(os.path.isfile(exe))
        if self.getDebugInfo() == "dwarf":
            self.assertFalse(os.path.isdir(dsym),
                             'testing DWARF, but .dSYM present')
        if self.getDebugInfo() == "dsym":
            self.assertTrue(os.path.isdir(dsym),
                            '.dSYM is missing in dsym config')

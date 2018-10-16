# TestREPLImportDispatch.py
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

from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import unittest2
import os

class TestREPLImportDispatch(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    @decorators.swiftTest
    @skipIf(debug_info=no_match(["dwarf"]))
    @decorators.add_test_categories(["swiftpr"])
    def test(self):
        self.build()
        self.assertTrue(os.path.isfile(self.getBuildArtifact("a.out")))
        # We're good! The test is part of the build.

if __name__ == '__main__':
    import atexit
    unittest2.main()

"""
Add a test to verify our test instance returns something non-None for
an id(). Other parts of the test running infrastructure are counting on this.
"""

from __future__ import print_function
from lldbsuite.test.lldbtest import TestBase

class TestIdTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)

    def test_id_exists(self):
        self.assertIsNotNone(self.id(), "Test instance should have an id()")


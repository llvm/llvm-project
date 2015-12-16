"""Tests that a FAIL is detected by the testbot."""

from __future__ import print_function

import lldbsuite.test.lldbtest as lldbtest


class FailTestCase(lldbtest.TestBase):
    """Forces test failure."""
    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def test_buildbot_catches_failure(self):
        """Issues a failing test assertion."""
        self.assertTrue(
            False,
            "This will always fail, buildbot should flag this.")

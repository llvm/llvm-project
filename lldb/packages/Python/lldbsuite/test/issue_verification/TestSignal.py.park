"""Tests that an exceptional exit is detected by the testbot."""

from __future__ import print_function

import os
import signal
import time

import lldbsuite.test.lldbtest as lldbtest


class ExceptionalExitTestCase(lldbtest.TestBase):
    """Forces exceptional exit."""
    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @lldbtest.skipIfWindows
    def test_buildbot_catches_exceptional_exit(self):
        """Force process to die with exceptional exit."""

        # Sleep for a couple seconds
        try:
            time.sleep(5)
        except:
            pass

        os.kill(os.getpid(), signal.SIGKILL)

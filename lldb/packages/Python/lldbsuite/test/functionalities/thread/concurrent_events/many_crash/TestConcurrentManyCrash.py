from __future__ import print_function

import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentManyCrash(ConcurrentEventsBase):

    mydir = ConcurrentEventsBase.compute_mydir(__file__)

    @unittest2.skipIf(
        TestBase.skipLongRunningTest(),
        "Skip this long running test")
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @expectedFailureAll(triple='^mips')
    def test_many_crash(self):
        """Test 100 threads that cause a segfault."""
        self.build(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_crash_threads=100)

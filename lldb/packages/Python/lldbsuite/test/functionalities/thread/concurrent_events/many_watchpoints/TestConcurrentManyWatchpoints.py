from __future__ import print_function

import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentManyWatchpoints(ConcurrentEventsBase):

    mydir = ConcurrentEventsBase.compute_mydir(__file__)

    @unittest2.skipIf(
        TestBase.skipLongRunningTest(),
        "Skip this long running test")
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple='^mips')
    @add_test_categories(["watchpoint"])
    def test(self):
        """Test 100 watchpoints from 100 threads."""
        self.build(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=100)

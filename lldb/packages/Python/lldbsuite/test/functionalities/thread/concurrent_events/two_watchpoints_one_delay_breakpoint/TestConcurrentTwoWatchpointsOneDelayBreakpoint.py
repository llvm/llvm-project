from __future__ import print_function

import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentTwoWatchpointsOneDelayBreakpoint(ConcurrentEventsBase):

    mydir = ConcurrentEventsBase.compute_mydir(__file__)

    @skipIfFreeBSD  # timing out on buildbot
    @skipIfRemoteDueToDeadlock
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @expectedFailureAll(triple='^mips')
    def test_two_watchpoints_one_delay_breakpoint(self):
        """Test two threads that trigger a watchpoint and one (1 second delay) breakpoint thread. """
        self.build(dictionary=self.getBuildFlags())
        self.do_thread_actions(
            num_watchpoint_threads=2,
            num_delay_breakpoint_threads=1)

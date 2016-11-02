from __future__ import print_function

import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentDelayedCrashWithBreakpointWatchpoint(ConcurrentEventsBase):

    mydir = ConcurrentEventsBase.compute_mydir(__file__)

    @skipIfFreeBSD  # timing out on buildbot
    @skipIfRemoteDueToDeadlock
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @expectedFailureAll(triple='^mips')
    def test_delayed_crash_with_breakpoint_watchpoint(self):
        """ Test a thread with a delayed crash while other threads hit a watchpoint and a breakpoint. """
        self.build(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_delay_crash_threads=1,
                               num_breakpoint_threads=1,
                               num_watchpoint_threads=1)

from __future__ import print_function

import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentTwoBreakpointsOneWatchpoint(ConcurrentEventsBase):

    mydir = ConcurrentEventsBase.compute_mydir(__file__)

    @skipIfFreeBSD # timing out on buildbot
    @skipIfRemoteDueToDeadlock
    @expectedFailureAll(triple = '^mips') # Atomic sequences are not supported yet for MIPS in LLDB.
    def test_two_breakpoints_one_watchpoint(self):
        """Test two threads that trigger a breakpoint and one watchpoint thread. """
        self.build(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=2, num_watchpoint_threads=1)



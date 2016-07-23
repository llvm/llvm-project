from __future__ import print_function

import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentTwoWatchpointsOneSignal(ConcurrentEventsBase):

    mydir = ConcurrentEventsBase.compute_mydir(__file__)

    @skipIfFreeBSD # timing out on buildbot
    @skipIfRemoteDueToDeadlock
    @expectedFailureAll(triple = '^mips') # Atomic sequences are not supported yet for MIPS in LLDB.
    def test_two_watchpoints_one_signal(self):
        """Test two threads that trigger a watchpoint and one signal thread. """
        self.build(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_watchpoint_threads=2, num_signal_threads=1)



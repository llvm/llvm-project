from __future__ import print_function

import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentCrashWithWatchpoint(ConcurrentEventsBase):

    mydir = ConcurrentEventsBase.compute_mydir(__file__)

    @skipIfFreeBSD # timing out on buildbot
    @skipIfRemoteDueToDeadlock
    @expectedFailureAll(triple = '^mips') # Atomic sequences are not supported yet for MIPS in LLDB.
    def test_crash_with_watchpoint(self):
        """ Test a thread that crashes while another thread hits a watchpoint."""
        self.build(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_crash_threads=1, num_watchpoint_threads=1)



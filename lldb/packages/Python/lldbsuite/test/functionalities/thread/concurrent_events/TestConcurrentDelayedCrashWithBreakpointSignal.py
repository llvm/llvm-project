from __future__ import print_function

import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentDelayedCrashWithBreakpointSignal(ConcurrentEventsBase):

    mydir = ConcurrentEventsBase.compute_mydir(__file__)

    @skipIfFreeBSD  # timing out on buildbot
    # Atomic sequences are not supported yet for MIPS in LLDB.
    @skipIf(triple='^mips')
    @expectedFailureNetBSD
    def test(self):
        """ Test a thread with a delayed crash while other threads generate a signal and hit a breakpoint. """
        self.build(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_delay_crash_threads=1,
                               num_breakpoint_threads=1,
                               num_signal_threads=1)

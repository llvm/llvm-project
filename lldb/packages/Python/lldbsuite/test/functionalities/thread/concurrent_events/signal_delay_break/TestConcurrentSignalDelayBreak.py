from __future__ import print_function

import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentSignalDelayBreak(ConcurrentEventsBase):

    mydir = ConcurrentEventsBase.compute_mydir(__file__)

    @skipIfFreeBSD # timing out on buildbot
    @expectedFailureAll(triple = '^mips') # Atomic sequences are not supported yet for MIPS in LLDB.
    def test_signal_delay_break(self):
        """Test signal and a (1 second delay) breakpoint in multiple threads."""
        self.build(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_delay_breakpoint_threads=1, num_signal_threads=1)




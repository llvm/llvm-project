from __future__ import print_function

import unittest2

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentManyBreakpoints(ConcurrentEventsBase):

    mydir = ConcurrentEventsBase.compute_mydir(__file__)

    @unittest2.skipIf(TestBase.skipLongRunningTest(), "Skip this long running test")
    @expectedFailureAll(triple = '^mips') # Atomic sequences are not supported yet for MIPS in LLDB.
    def test_many_breakpoints(self):
        """Test 100 breakpoints from 100 threads."""
        self.build(dictionary=self.getBuildFlags())
        self.do_thread_actions(num_breakpoint_threads=100)



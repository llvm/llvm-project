"""Test that the Objective-C syntax for dictionary/array literals and indexing works"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCNewSyntaxTest(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def runToBreakpoint(self):
        self.build()
        self.target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, '// Set breakpoint 0 here.', lldb.SBFileSpec(
                'main.m', False))

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=['stopped', 'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect(
            "breakpoint list -f",
            BREAKPOINT_HIT_ONCE,
            substrs=[' resolved, hit count = 1'])

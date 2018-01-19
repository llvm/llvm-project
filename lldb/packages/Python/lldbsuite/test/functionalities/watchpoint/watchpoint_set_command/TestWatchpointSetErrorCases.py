"""
Test error cases for the 'watchpoint set' command to make sure it errors out when necessary.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class WatchpointSetErrorTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.cpp'
        # Find the line number to break inside main().
        self.line = line_number(
            self.source, '// Set break point at this line.')
        # Build dictionary to have unique executable names for each test
        # method.

    def test_error_cases_with_watchpoint_set(self):
        """Test error cases with the 'watchpoint set' command."""
        self.build()
        self.setTearDownCleanup()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        lldbutil.run_break_set_by_file_and_line(
            self, None, self.line, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to the breakpoint.
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # Try some error conditions:

        # 'watchpoint set' is now a multiword command.
        self.expect("watchpoint set",
                    substrs=['The following subcommands are supported:',
                             'expression',
                             'variable'])
        self.runCmd("watchpoint set variable -w read_write", check=False)

        # 'watchpoint set expression' with '-w' or '-s' specified now needs
        # an option terminator and a raw expression after that.
        self.expect("watchpoint set expression -w write --", error=True,
                    startstr='error: ')

        # It's an error if the expression did not evaluate to an address.
        self.expect(
            "watchpoint set expression MyAggregateDataType",
            error=True,
            startstr='error: expression did not evaluate to an address')

        # Wrong size parameter is an error.
        self.expect("watchpoint set variable -s -128", error=True,
                    substrs=['invalid enumeration value'])

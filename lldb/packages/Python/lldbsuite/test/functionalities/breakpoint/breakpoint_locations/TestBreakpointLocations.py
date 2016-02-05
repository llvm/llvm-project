"""
Test breakpoint commands for a breakpoint ID with multiple locations.
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class BreakpointLocationsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureWindows("llvm.org/pr24528")
    @expectedFailureAll(oslist=["linux"], compiler="clang", compiler_version=[">=", "3.8"], archs=["i386"], debug_info="dwo")
    def test(self):
        """Test breakpoint enable/disable for a breakpoint ID with multiple locations."""
        self.build()
        self.breakpoint_locations_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    def breakpoint_locations_test(self):
        """Test breakpoint enable/disable for a breakpoint ID with multiple locations."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint with 3 locations.
        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, num_expected_locations=3)

        # The breakpoint list should show 3 locations.
        self.expect("breakpoint list -f", "Breakpoint locations shown correctly",
            substrs = ["1: file = 'main.c', line = %d, exact_match = 0, locations = 3" % self.line],
            patterns = ["where = a.out`func_inlined .+unresolved, hit count = 0",
                        "where = a.out`main .+\[inlined\].+unresolved, hit count = 0"])

        # The 'breakpoint disable 3.*' command should fail gracefully.
        self.expect("breakpoint disable 3.*",
                    "Disabling an invalid breakpoint should fail gracefully",
                    error=True,
            startstr = "error: '3' is not a valid breakpoint ID.")

        # The 'breakpoint disable 1.*' command should disable all 3 locations.
        self.expect("breakpoint disable 1.*", "All 3 breakpoint locatons disabled correctly",
            startstr = "3 breakpoints disabled.")

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should not stopped on any breakpoint at all.
        self.expect("process status", "No stopping on any disabled breakpoint",
            patterns = ["^Process [0-9]+ exited with status = 0"])

        # The 'breakpoint enable 1.*' command should enable all 3 breakpoints.
        self.expect("breakpoint enable 1.*", "All 3 breakpoint locatons enabled correctly",
            startstr = "3 breakpoints enabled.")

        # The 'breakpoint disable 1.1' command should disable 1 location.
        self.expect("breakpoint disable 1.1", "1 breakpoint locatons disabled correctly",
            startstr = "1 breakpoints disabled.")

        # Run the program againt.  We should stop on the two breakpoint locations.
        self.runCmd("run", RUN_SUCCEEDED)

        # Stopped once.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ["stop reason = breakpoint 1."])

        # Continue the program, there should be another stop.
        self.runCmd("process continue")

        # Stopped again.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ["stop reason = breakpoint 1."])

        # At this point, 1.1 has a hit count of 0 and the other a hit count of 1".
        self.expect("breakpoint list -f", "The breakpoints should report correct hit counts",
            patterns = ["1\.1: .+ unresolved, hit count = 0 +Options: disabled",
                        "1\.2: .+ resolved, hit count = 1",
                        "1\.3: .+ resolved, hit count = 1"])

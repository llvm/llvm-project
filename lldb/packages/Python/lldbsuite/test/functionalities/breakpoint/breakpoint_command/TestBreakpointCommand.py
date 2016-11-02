"""
Test lldb breakpoint command add/list/delete.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BreakpointCommandTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @classmethod
    def classCleanup(cls):
        """Cleanup the test byproduct of breakpoint_command_sequence(self)."""
        cls.RemoveTempFile("output.txt")
        cls.RemoveTempFile("output2.txt")

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24528")
    def test(self):
        """Test a sequence of breakpoint command add, list, and delete."""
        self.build()
        self.breakpoint_command_sequence()
        self.breakpoint_command_script_parameters()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')
        # disable "There is a running process, kill it and restart?" prompt
        self.runCmd("settings set auto-confirm true")
        self.addTearDownHook(
            lambda: self.runCmd("settings clear auto-confirm"))

    def breakpoint_command_sequence(self):
        """Test a sequence of breakpoint command add, list, and delete."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add three breakpoints on the same line.  The first time we don't specify the file,
        # since the default file is the one containing main:
        lldbutil.run_break_set_by_file_and_line(
            self, None, self.line, num_expected_locations=1, loc_exact=True)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True)
        # Breakpoint 4 - set at the same location as breakpoint 1 to test
        # setting breakpoint commands on two breakpoints at a time
        lldbutil.run_break_set_by_file_and_line(
            self, None, self.line, num_expected_locations=1, loc_exact=True)

        # Now add callbacks for the breakpoints just created.
        self.runCmd(
            "breakpoint command add -s command -o 'frame variable --show-types --scope' 1 4")
        self.runCmd(
            "breakpoint command add -s python -o 'here = open(\"output.txt\", \"w\"); here.write(\"lldb\\n\"); here.close()' 2")
        self.runCmd(
            "breakpoint command add --python-function bktptcmd.function 3")

        # Check that the breakpoint commands are correctly set.

        # The breakpoint list now only contains breakpoint 1.
        self.expect(
            "breakpoint list", "Breakpoints 1 & 2 created", substrs=[
                "2: file = 'main.c', line = %d, exact_match = 0, locations = 1" %
                self.line], patterns=[
                "1: file = '.*main.c', line = %d, exact_match = 0, locations = 1" %
                self.line])

        self.expect(
            "breakpoint list -f",
            "Breakpoints 1 & 2 created",
            substrs=[
                "2: file = 'main.c', line = %d, exact_match = 0, locations = 1" %
                self.line],
            patterns=[
                "1: file = '.*main.c', line = %d, exact_match = 0, locations = 1" %
                self.line,
                "1.1: .+at main.c:%d, .+unresolved, hit count = 0" %
                self.line,
                "2.1: .+at main.c:%d, .+unresolved, hit count = 0" %
                self.line])

        self.expect("breakpoint command list 1", "Breakpoint 1 command ok",
                    substrs=["Breakpoint commands:",
                             "frame variable --show-types --scope"])
        self.expect("breakpoint command list 2", "Breakpoint 2 command ok",
                    substrs=["Breakpoint commands:",
                             "here = open",
                             "here.write",
                             "here.close()"])
        self.expect("breakpoint command list 3", "Breakpoint 3 command ok",
                    substrs=["Breakpoint commands:",
                             "bktptcmd.function(frame, bp_loc, internal_dict)"])

        self.expect("breakpoint command list 4", "Breakpoint 4 command ok",
                    substrs=["Breakpoint commands:",
                             "frame variable --show-types --scope"])

        self.runCmd("breakpoint delete 4")

        self.runCmd("command script import --allow-reload ./bktptcmd.py")

        # Next lets try some other breakpoint kinds.  First break with a regular expression
        # and then specify only one file.  The first time we should get two locations,
        # the second time only one:

        lldbutil.run_break_set_by_regexp(
            self, r"._MyFunction", num_expected_locations=2)

        lldbutil.run_break_set_by_regexp(
            self,
            r"._MyFunction",
            extra_options="-f a.c",
            num_expected_locations=1)

        lldbutil.run_break_set_by_regexp(
            self,
            r"._MyFunction",
            extra_options="-f a.c -f b.c",
            num_expected_locations=2)

        # Now try a source regex breakpoint:
        lldbutil.run_break_set_by_source_regexp(
            self,
            r"is about to return [12]0",
            extra_options="-f a.c -f b.c",
            num_expected_locations=2)

        lldbutil.run_break_set_by_source_regexp(
            self,
            r"is about to return [12]0",
            extra_options="-f a.c",
            num_expected_locations=1)

        # Run the program.  Remove 'output.txt' if it exists.
        self.RemoveTempFile("output.txt")
        self.RemoveTempFile("output2.txt")
        self.runCmd("run", RUN_SUCCEEDED)

        # Check that the file 'output.txt' exists and contains the string
        # "lldb".

        # The 'output.txt' file should now exist.
        self.assertTrue(
            os.path.isfile("output.txt"),
            "'output.txt' exists due to breakpoint command for breakpoint 2.")
        self.assertTrue(
            os.path.isfile("output2.txt"),
            "'output2.txt' exists due to breakpoint command for breakpoint 3.")

        # Read the output file produced by running the program.
        with open('output.txt', 'r') as f:
            output = f.read()

        self.expect(
            output,
            "File 'output.txt' and the content matches",
            exe=False,
            startstr="lldb")

        with open('output2.txt', 'r') as f:
            output = f.read()

        self.expect(
            output,
            "File 'output2.txt' and the content matches",
            exe=False,
            startstr="lldb")

        # Finish the program.
        self.runCmd("process continue")

        # Remove the breakpoint command associated with breakpoint 1.
        self.runCmd("breakpoint command delete 1")

        # Remove breakpoint 2.
        self.runCmd("breakpoint delete 2")

        self.expect(
            "breakpoint command list 1",
            startstr="Breakpoint 1 does not have an associated command.")
        self.expect(
            "breakpoint command list 2",
            error=True,
            startstr="error: '2' is not a currently valid breakpoint ID.")

        # The breakpoint list now only contains breakpoint 1.
        self.expect(
            "breakpoint list -f",
            "Breakpoint 1 exists",
            patterns=[
                "1: file = '.*main.c', line = %d, exact_match = 0, locations = 1, resolved = 1" %
                self.line,
                "hit count = 1"])

        # Not breakpoint 2.
        self.expect(
            "breakpoint list -f",
            "No more breakpoint 2",
            matching=False,
            substrs=[
                "2: file = 'main.c', line = %d, exact_match = 0, locations = 1, resolved = 1" %
                self.line])

        # Run the program again, with breakpoint 1 remaining.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to breakpoint 1.

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 2.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_TWICE,
                    substrs=['resolved, hit count = 2'])

    def breakpoint_command_script_parameters(self):
        """Test that the frame and breakpoint location are being properly passed to the script breakpoint command function."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True)

        # Now add callbacks for the breakpoints just created.
        self.runCmd("breakpoint command add -s python -o 'here = open(\"output-2.txt\", \"w\"); here.write(str(frame) + \"\\n\"); here.write(str(bp_loc) + \"\\n\"); here.close()' 1")

        # Remove 'output-2.txt' if it already exists.

        if (os.path.exists('output-2.txt')):
            os.remove('output-2.txt')

        # Run program, hit breakpoint, and hopefully write out new version of
        # 'output-2.txt'
        self.runCmd("run", RUN_SUCCEEDED)

        # Check that the file 'output.txt' exists and contains the string
        # "lldb".

        # The 'output-2.txt' file should now exist.
        self.assertTrue(
            os.path.isfile("output-2.txt"),
            "'output-2.txt' exists due to breakpoint command for breakpoint 1.")

        # Read the output file produced by running the program.
        with open('output-2.txt', 'r') as f:
            output = f.read()

        self.expect(
            output,
            "File 'output-2.txt' and the content matches",
            exe=False,
            startstr="frame #0:",
            patterns=["1.* where = .*main .* resolved, hit count = 1"])

        # Now remove 'output-2.txt'
        os.remove('output-2.txt')

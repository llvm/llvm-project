"""Test the RunCommandInterpreter API."""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

class CommandRunInterpreterAPICase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

        self.stdin_path = self.getBuildArtifact("stdin.txt")

        with open(self.stdin_path, 'w') as input_handle:
            input_handle.write("nonexistingcommand\nquit")

        with open(self.stdin_path, 'r') as input_handle:
            self.dbg.SetInputFileHandle(input_handle, False)

        # No need to track the output
        devnull = open(os.devnull, 'w')
        self.dbg.SetOutputFileHandle(devnull, False)
        self.dbg.SetErrorFileHandle(devnull, False)

    @add_test_categories(['pyapi'])
    def test_run_session_with_error_and_quit(self):
        """Run non-existing and quit command returns appropriate values"""

        n_errors, quit_requested, has_crashed = self.dbg.RunCommandInterpreter(
                True, False, lldb.SBCommandInterpreterRunOptions(), 0, False,
                False)

        self.assertGreater(n_errors, 0)
        self.assertTrue(quit_requested)
        self.assertFalse(has_crashed)

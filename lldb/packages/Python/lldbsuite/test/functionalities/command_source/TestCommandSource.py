"""
Test that lldb command "command source" works correctly.

See also http://llvm.org/viewvc/llvm-project?view=rev&revision=109673.
"""

from __future__ import print_function



import os, sys
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class CommandSourceTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_command_source(self):
        """Test that lldb command "command source" works correctly."""

        # Sourcing .lldb in the current working directory, which in turn imports
        # the "my" package that defines the date() function.
        self.runCmd("command source .lldb")

        # Python should evaluate "my.date()" successfully.
        command_interpreter = self.dbg.GetCommandInterpreter()
        self.assertTrue(command_interpreter, VALID_COMMAND_INTERPRETER)
        result = lldb.SBCommandReturnObject()
        command_interpreter.HandleCommand("script my.date()", result)

        import datetime
        self.expect(result.GetOutput(), "script my.date() runs successfully",
                    exe=False,
            substrs = [str(datetime.date.today())])

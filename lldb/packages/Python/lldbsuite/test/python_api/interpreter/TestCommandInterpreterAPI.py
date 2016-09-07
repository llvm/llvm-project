"""Test the SBCommandInterpreter APIs."""

from __future__ import print_function


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CommandInterpreterAPICase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break on inside main.cpp.
        self.line = line_number('main.c', 'Hello world.')

    @add_test_categories(['pyapi'])
    def test_with_process_launch_api(self):
        """Test the SBCommandInterpreter APIs."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Retrieve the associated command interpreter from our debugger.
        ci = self.dbg.GetCommandInterpreter()
        self.assertTrue(ci, VALID_COMMAND_INTERPRETER)

        # Exercise some APIs....

        self.assertTrue(ci.HasCommands())
        self.assertTrue(ci.HasAliases())
        self.assertTrue(ci.HasAliasOptions())
        self.assertTrue(ci.CommandExists("breakpoint"))
        self.assertTrue(ci.CommandExists("target"))
        self.assertTrue(ci.CommandExists("platform"))
        self.assertTrue(ci.AliasExists("file"))
        self.assertTrue(ci.AliasExists("run"))
        self.assertTrue(ci.AliasExists("bt"))

        res = lldb.SBCommandReturnObject()
        ci.HandleCommand("breakpoint set -f main.c -l %d" % self.line, res)
        self.assertTrue(res.Succeeded())
        ci.HandleCommand("process launch", res)
        self.assertTrue(res.Succeeded())

        # Boundary conditions should not crash lldb!
        self.assertFalse(ci.CommandExists(None))
        self.assertFalse(ci.AliasExists(None))
        ci.HandleCommand(None, res)
        self.assertFalse(res.Succeeded())
        res.AppendMessage("Just appended a message.")
        res.AppendMessage(None)
        if self.TraceOn():
            print(res)

        process = ci.GetProcess()
        self.assertTrue(process)

        import lldbsuite.test.lldbutil as lldbutil
        if process.GetState() != lldb.eStateStopped:
            self.fail("Process should be in the 'stopped' state, "
                      "instead the actual state is: '%s'" %
                      lldbutil.state_type_to_str(process.GetState()))

        if self.TraceOn():
            lldbutil.print_stacktraces(process)

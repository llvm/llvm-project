"""
Test that breakpoints work in a DLL
"""

from __future__ import print_function

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessWindows
class WindowsDLLTestCase(TestBase):
    def setUP(self):
        TestBase.setUp(self)
        self.build()

    def test_dll_linking(self):
        """test that the debugger works with DLLs"""

        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target and target.IsValid(), "Target is valid")

        self.runCmd("breakpoint set --file main.c --line 16")
        self.runCmd("breakpoint set --file dllfunc.c --line 18")

        process = target.LaunchSimple(None, None, self.get_process_working_directory())

        self.expect("p x", "16")
        self.runCmd("thread step-out")
        self.expect("p x", "16")
        self.expect("thread step-in")
        self.expect("thread step-in")
        self.expect("p n", "8")
        self.runCmd("c")
        self.expect("p x", "64")
        self.runCmd("breakpoint delete 2")
        self.runCmd("c")

        self.assertEqual(process.GetExitStatus(), 336, PROCESS_EXITED)

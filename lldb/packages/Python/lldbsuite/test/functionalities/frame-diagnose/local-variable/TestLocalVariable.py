"""
Test the output of `frame diagnose` for dereferencing a local variable
"""

from __future__ import print_function

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestLocalVariable(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    def test_local_variable(self):
        TestBase.setUp(self)
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        self.runCmd("run", RUN_SUCCEEDED)
        self.expect("thread list", "Thread should be stopped",
            substrs = ['stopped'])
        self.expect("frame diagnose", "Crash diagnosis was accurate", "myInt") 

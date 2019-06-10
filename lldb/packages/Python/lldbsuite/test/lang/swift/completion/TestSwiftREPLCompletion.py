from __future__ import print_function
import pexpect
import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestSwiftREPLCompletion(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    def test_repl_completion(self):
        prompt = "Welcome to"
        child = pexpect.spawn('%s --repl' % (lldbtest_config.lldbExec))
        # Assign to make sure the sessions are killed during teardown
        self.child = child
        # Send a <TAB> and make sure we don't crash.
        child.sendline("import Foundatio\t")
        child.sendline("print(NSString(\"patatino\"))")
        child.expect("patatino")

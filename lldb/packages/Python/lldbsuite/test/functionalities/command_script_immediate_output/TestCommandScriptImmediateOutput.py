"""
Test that LLDB correctly allows scripted commands to set an immediate output file
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import *

class CommandScriptImmediateOutputTestCase (PExpectTest):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        PExpectTest.setUp(self)

    @skipIfRemote # test not remote-ready llvm.org/pr24813
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_command_script_immediate_output (self):
        """Test that LLDB correctly allows scripted commands to set an immediate output file."""
        self.launch(timeout=5)

        script = os.path.join(os.getcwd(), 'custom_command.py')
        prompt = "(lldb)"
        
        self.sendline('command script import %s' % script, patterns=[prompt])
        self.sendline('command script add -f custom_command.command_function mycommand', patterns=[prompt])
        self.sendline('mycommand', patterns='this is a test string, just a test string')
        self.sendline('command script delete mycommand', patterns=[prompt])
        self.quit(gracefully=False)

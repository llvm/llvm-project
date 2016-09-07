"""
Test that LLDB correctly allows scripted commands to set an immediate output file
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import *
from lldbsuite.test import lldbutil


class CommandScriptImmediateOutputTestCase (PExpectTest):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        PExpectTest.setUp(self)

    @skipIfRemote  # test not remote-ready llvm.org/pr24813
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    @expectedFailureAll(oslist=["freebsd"], bugnumber="llvm.org/pr26139")
    def test_command_script_immediate_output_console(self):
        """Test that LLDB correctly allows scripted commands to set immediate output to the console."""
        self.launch(timeout=10)

        script = os.path.join(os.getcwd(), 'custom_command.py')
        prompt = "\(lldb\) "

        self.sendline('command script import %s' % script, patterns=[prompt])
        self.sendline(
            'command script add -f custom_command.command_function mycommand',
            patterns=[prompt])
        self.sendline(
            'mycommand',
            patterns='this is a test string, just a test string')
        self.sendline('command script delete mycommand', patterns=[prompt])
        self.quit(gracefully=False)

    @skipIfRemote  # test not remote-ready llvm.org/pr24813
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    @expectedFailureAll(oslist=["freebsd"], bugnumber="llvm.org/pr26139")
    def test_command_script_immediate_output_file(self):
        """Test that LLDB correctly allows scripted commands to set immediate output to a file."""
        self.launch(timeout=10)

        test_files = {os.path.join(os.getcwd(), 'read.txt'): 'r',
                      os.path.join(os.getcwd(), 'write.txt'): 'w',
                      os.path.join(os.getcwd(), 'append.txt'): 'a',
                      os.path.join(os.getcwd(), 'write_plus.txt'): 'w+',
                      os.path.join(os.getcwd(), 'read_plus.txt'): 'r+',
                      os.path.join(os.getcwd(), 'append_plus.txt'): 'a+'}

        starter_string = 'Starter Garbage\n'
        write_string = 'writing to file with mode: '

        for path, mode in test_files.iteritems():
            with open(path, 'w+') as init:
                init.write(starter_string)

        script = os.path.join(os.getcwd(), 'custom_command.py')
        prompt = "\(lldb\) "

        self.sendline('command script import %s' % script, patterns=[prompt])

        self.sendline(
            'command script add -f custom_command.write_file mywrite',
            patterns=[prompt])
        for path, mode in test_files.iteritems():
            command = 'mywrite "' + path + '" ' + mode

            self.sendline(command, patterns=[prompt])

        self.sendline('command script delete mywrite', patterns=[prompt])

        self.quit(gracefully=False)

        for path, mode in test_files.iteritems():
            with open(path, 'r') as result:
                if mode in ['r', 'a', 'a+']:
                    self.assertEquals(result.readline(), starter_string)
                if mode in ['w', 'w+', 'r+', 'a', 'a+']:
                    self.assertEquals(
                        result.readline(), write_string + mode + '\n')

            self.assertTrue(os.path.isfile(path))
            os.remove(path)

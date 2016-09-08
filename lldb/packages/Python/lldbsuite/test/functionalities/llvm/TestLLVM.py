"""
Test lldb 'commands regex' command which allows the user to create a regular expression command.
"""

from __future__ import print_function


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestHomeDirectory(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(oslist=["windows"])
    @no_debug_info_test
    def test_tilde_home_directory(self):
        """Test that we can resolve "~/" in paths correctly.
        When a path starts with "~/", we use llvm::sys::path::home_directory() to
        resolve the home directory. This currently relies on "HOME" being set in the
        environment. While this is usually set, we can't rely upon that. We might
        eventually get a fix into llvm::sys::path::home_directory() so it doesn't rely
        on having to have an environment variable set, but until then we have work around
        code in FileSpec::ResolveUsername (llvm::SmallVectorImpl<char> &path) to ensure
        this always works. This test tests that we get the correct answer for with and
        without "HOME" being set in the environment."""
        import pexpect
        prompt = "(lldb) "

        child = pexpect.spawn(
            '%s --no-use-colors %s' %
            (lldbtest_config.lldbExec, self.lldbOption))
        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout
        # So that the spawned lldb session gets shutdown durng teardown.
        self.child = child

        # Resolve "~/." to the full path of our home directory + "/."
        if 'HOME' in os.environ:
            home_dir = os.environ['HOME']
            if self.TraceOn():
                print("home directory is: '%s'" % (home_dir))
            if os.path.exists(home_dir):
                home_dir_slash_dot = home_dir + '/.'
                child.expect_exact(prompt)
                child.sendline('''script str(lldb.SBFileSpec("~/.", True))''')
                child.expect_exact(home_dir)
                child.expect_exact(prompt)
                child.sendline(
                    '''script import os; os.unsetenv('HOME'); str(lldb.SBFileSpec("~/", True))''')
                child.expect_exact(home_dir)
                child.expect_exact(prompt)
            elif self.TraceOn():
                print(
                    '''home directory "%s" doesn't exist, skipping home directory test''' %
                    (home_dir))
        elif self.TraceOn():
            print('"HOME" not in environment, skipping home directory test')

        child.sendline('quit')

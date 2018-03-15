"""
Test the lldb command line takes a filename with single quote chars.
"""

from __future__ import print_function


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import six

class SingleQuoteInCommandLineTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    myexe = "path with '09/a.out"

    @classmethod
    def classCleanup(cls):
        """Cleanup the test byproducts."""
        try:
            os.remove("child_send.txt")
            os.remove("child_read.txt")
            os.remove(cls.myexe)
        except:
            pass

    @expectedFailureAll(
        hostoslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    @no_debug_info_test
    def test_lldb_invocation_with_single_quote_in_filename(self):
        """Test that 'lldb my_file_name' works where my_file_name is a string with a single quote char in it."""
        import pexpect
        self.buildDefault()
        lldbutil.mkdir_p(self.getBuildArtifact("path with '09"))
        system([["cp",
                 self.getBuildArtifact("a.out"),
                 "\"%s\"" % self.getBuildArtifact(self.myexe)]])

        # The default lldb prompt.
        prompt = "(lldb) "

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn(
            '%s %s "%s"' %
            (lldbtest_config.lldbExec, self.lldbOption,
             self.getBuildArtifact(self.myexe)))
        child = self.child
        child.setecho(True)
        child.logfile_send = send = six.StringIO()
        child.logfile_read = read = six.StringIO()
        child.expect_exact(prompt)

        child.send("help watchpoint")
        child.sendline('')
        child.expect_exact(prompt)

        # Now that the necessary logging is done, restore logfile to None to
        # stop further logging.
        child.logfile_send = None
        child.logfile_read = None

        if self.TraceOn():
            print("\n\nContents of send")
            print(send.getvalue())
            print("\n\nContents of read")
            print(read.getvalue())

        self.expect(read.getvalue(), exe=False,
                    substrs=["Current executable set to"])

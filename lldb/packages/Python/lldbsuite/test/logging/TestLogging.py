"""
Test lldb logging.  This test just makes sure logging doesn't crash, and produces some output.
"""

from __future__ import print_function


import os
import time
import string
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LogTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        super(LogTestCase, self).setUp()
        self.log_file = self.getBuildArtifact("log-file.txt")

    def test(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.expect("file " + exe,
                    patterns=["Current executable set to .*a.out"])

        log_file = os.path.join(self.getBuildDir(), "lldb-commands-log.txt")

        if (os.path.exists(log_file)):
            os.remove(log_file)

        # By default, Debugger::EnableLog() will set log options to
        # PREPEND_THREAD_NAME + OPTION_THREADSAFE. We don't want the
        # threadnames here, so we enable just threadsafe (-t).
        self.runCmd("log enable -t -f '%s' lldb commands" % (log_file))

        self.runCmd("command alias bp breakpoint")

        self.runCmd("bp set -n main")

        self.runCmd("bp l")

        self.runCmd("log disable lldb")

        self.assertTrue(os.path.isfile(log_file))

        f = open(log_file)
        log_lines = f.readlines()
        f.close()
        os.remove(log_file)

        self.assertGreater(
            len(log_lines),
            0,
            "Something was written to the log file.")

    # Check that lldb truncates its log files
    def test_log_truncate(self):
        # put something in our log file
        with open(self.log_file, "w") as f:
            for i in range(1, 1000):
                f.write("bacon\n")

        self.runCmd("log enable -t -f '%s' lldb commands" % self.log_file)
        self.runCmd("help log")
        self.runCmd("log disable lldb")

        self.assertTrue(os.path.isfile(self.log_file))
        with open(self.log_file, "r") as f:
            contents = f.read()

        # check that it got removed
        self.assertEquals(contents.find("bacon"), -1)

    # Check that lldb can append to a log file
    def test_log_append(self):
        # put something in our log file
        with open(self.log_file, "w") as f:
            f.write("bacon\n")

        self.runCmd( "log enable -t -a -f '%s' lldb commands" % self.log_file)
        self.runCmd("help log")
        self.runCmd("log disable lldb")

        self.assertTrue(os.path.isfile(self.log_file))
        with open(self.log_file, "r") as f:
            contents = f.read()

        # check that it is still there
        self.assertEquals(contents.find("bacon"), 0)

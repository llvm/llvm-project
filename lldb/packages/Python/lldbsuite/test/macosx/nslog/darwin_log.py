"""
Base class for DarwinLog tests.
"""

from __future__ import print_function

import lldb
import os
import pexpect
import re
import sys

from lldbsuite.test import decorators
from lldbsuite.test import lldbtest
from lldbsuite.test import lldbtest_config
from lldbsuite.test import lldbutil


class DarwinLogTestBase(lldbtest.TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        super(DarwinLogTestBase, self).setUp()
        self.child = None
        self.child_prompt = '(lldb) '
        self.strict_sources = False

    def run_lldb_to_breakpoint(self, exe, source_file, line,
                               settings_commands=None):
        # Set self.child_prompt, which is "(lldb) ".
        prompt = self.child_prompt

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn('%s %s %s' % (lldbtest_config.lldbExec,
                                                 self.lldbOption, exe))
        child = self.child

        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        # Disable showing of source lines at our breakpoint.
        # This is necessary for the logging tests, because the very
        # text we want to match for output from the running inferior
        # will show up in the source as well.  We don't want the source
        # output to erroneously make a match with our expected output.
        self.runCmd("settings set stop-line-count-before 0")
        self.expect_prompt()
        self.runCmd("settings set stop-line-count-after 0")
        self.expect_prompt()

        # This setting is required to be set to true in order to get LLDB to set
        # the environment variable required by libtrace to have it respect our
        # 'plugin structured-data darwin-log source enable/disable' settings.
        # Without that, libtrace assumes we want all types of messages when
        # attaching with a debugger.
        setting = "strict-sources " + "true" if self.strict_sources else "false"
        self.runCmd(self.expand_darwinlog_settings_set_command(setting))
        self.expect_prompt()

        # Run any darwin-log settings commands now, before we enable logging.
        if settings_commands is not None:
            for setting_command in settings_commands:
                self.runCmd(
                    self.expand_darwinlog_settings_set_command(setting_command))
                self.expect_prompt()

        # child.expect_exact(prompt)
        child.sendline('breakpoint set -f %s -l %d' % (source_file, line))
        child.expect_exact(prompt)
        child.sendline('run')
        child.expect_exact(prompt)

        # Ensure we stopped at a breakpoint.
        self.runCmd("thread list")
        self.expect(re.compile(r"stop reason = breakpoint"))

        # Now we're ready to check if DarwinLog is available.
        if not self.darwin_log_available():
            self.skip("DarwinLog not available")

    def runCmd(self, cmd):
        self.child.sendline(cmd)

    def expect_prompt(self, exactly=True):
        self.expect(self.child_prompt, exactly=exactly)

    def expect(self, pattern, exactly=False, *args, **kwargs):
        if exactly:
            return self.child.expect_exact(pattern, *args, **kwargs)
        return self.child.expect(pattern, *args, **kwargs)

    def darwin_log_available(self):
        self.runCmd("plugin structured-data darwin-log status")
        self.expect(re.compile(r"Availability: ([\S]+)"))
        return self.child.match is not None and (
            self.child.match.group(1) == "available")

    @classmethod
    def expand_darwinlog_command(cls, command):
        return "plugin structured-data darwin-log " + command

    @classmethod
    def expand_darwinlog_settings_set_command(cls, command):
        return "settings set plugin.structured-data.darwin-log." + command

    def do_test(self, logging_setup_commands, expect_regexes=None,
                settings_commands=None):
        """Test that a single fall-through reject rule rejects all logging."""
        self.build(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)

        exe = os.path.join(os.getcwd(), self.exe_name)
        self.run_lldb_to_breakpoint(exe, self.source, self.line,
                                    settings_commands=settings_commands)
        self.expect_prompt()

        # Run each of the logging setup commands.
        for setup_command in logging_setup_commands:
            self.runCmd(self.expand_darwinlog_command(setup_command))
            self.expect_prompt()

        # Enable logging.
        self.runCmd(self.expand_darwinlog_command("enable"))
        self.expect_prompt()

        # Now go.
        self.runCmd("process continue")

        if expect_regexes is None:
            # Expect matching a log line or program exit.
            # Test methods determine which ones are valid.
            expect_regexes = (
                [re.compile(r"source-log-([^-]+)-(\S+)"),
                 re.compile(r"exited with status")
                ])
        self.expect(expect_regexes)


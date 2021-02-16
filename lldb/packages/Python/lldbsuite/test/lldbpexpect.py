from __future__ import absolute_import

# System modules
import os
import sys

# Third-party modules
import six

# LLDB Modules
import lldb
from .lldbtest import *
from . import lldbutil
from lldbsuite.test.decorators import *

@skipIfRemote
@skipIfWindows  # llvm.org/pr22274: need a pexpect replacement for windows
class PExpectTest(TestBase):

    NO_DEBUG_INFO_TESTCASE = True
    PROMPT = "(lldb) "

    def expect_prompt(self):
        self.child.expect_exact(self.PROMPT)

    def launch(self, executable=None, extra_args=None, timeout=30, dimensions=None):
        logfile = getattr(sys.stdout, 'buffer',
                            sys.stdout) if self.TraceOn() else None

        args = ['--no-lldbinit', '--no-use-colors']
        for cmd in self.setUpCommands():
            args += ['-O', cmd]
        if executable is not None:
            args += ['--file', executable]
        if extra_args is not None:
            args.extend(extra_args)

        env = dict(os.environ)
        env["TERM"]="vt100"

        import pexpect
        self.child = pexpect.spawn(
                lldbtest_config.lldbExec, args=args, logfile=logfile,
                timeout=timeout, dimensions=dimensions, env=env)
        self.expect_prompt()
        for cmd in self.setUpCommands():
            self.child.expect_exact(cmd)
            self.expect_prompt()
        if executable is not None:
            self.child.expect_exact("target create")
            self.child.expect_exact("Current executable set to")
            self.expect_prompt()

    def expect(self, cmd, substrs=None):
        self.assertNotIn('\n', cmd)
        # If 'substrs' is a string then this code would just check that every
        # character of the string is in the output.
        assert not isinstance(substrs, six.string_types), \
            "substrs must be a collection of strings"

        self.child.sendline(cmd)
        if substrs is not None:
            for s in substrs:
                self.child.expect_exact(s)
        self.expect_prompt()

    def quit(self, gracefully=True):
        self.child.sendeof()
        self.child.close(force=not gracefully)
        self.child = None

    def cursor_forward_escape_seq(self, chars_to_move):
        """
        Returns the escape sequence to move the cursor forward/right
        by a certain amount of characters.
        """
        return b"\x1b\[" + str(chars_to_move).encode("utf-8") + b"C"

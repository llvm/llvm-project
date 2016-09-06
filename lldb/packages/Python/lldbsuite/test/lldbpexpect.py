from __future__ import print_function
from __future__ import absolute_import



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil
# System modules
import os
import sys

# Third-party modules
import six

# LLDB Modules
import lldb
from .lldbtest import *
from . import lldbutil

if sys.platform.startswith('win32'):
    class PExpectTest(TestBase):
        pass
else:
    import pexpect

    class LLDBPExpectException(Exception):
      def __init__(self, kind, patterns, child):
        self.kind = kind
        self.patterns = patterns
        self.child = child
      
      def str_as_pattern(self, s):
        return '\n------\n%s\n------\n' % s
      
      def __str__(self):
        s = '%s error raised in pexpect.\n' % self.kind
        s += 'expecting patterns%sinstead received %s\n' % (self.str_as_pattern(self.patterns), self.str_as_pattern(self.child.buffer))
        s += 'stored buffer contents are %s\n' % self.str_as_pattern(self.child.before)
        s += 'inferior process (pid = %d) has%sbeen closed - exit status is %s' % (self.child.pid, ' ' if self.child.closed else ' not ', self.child.exitstatus)
        return s

    class PExpectTest(TestBase):
    
        mydir = TestBase.compute_mydir(__file__)

        def setUp(self):
            TestBase.setUp(self)

        def launchArgs(self):
            pass

        def launch(self, timeout=None):
            if timeout is None: timeout = 30
            logfile = sys.stdout if self.TraceOn() else None
            self.child = pexpect.spawn('%s --no-use-colors %s' % (lldbtest_config.lldbExec, self.launchArgs()), logfile=logfile)
            self.child.timeout = timeout
            self.timeout = timeout

        def expect(self, patterns=None, timeout=None, exact=None):
            if patterns is None: return None
            if timeout is None: timeout = self.timeout
            if exact is None: exact = False
            try:
              if exact:
                  return self.child.expect_exact(patterns, timeout=timeout)
              else:
                  return self.child.expect(patterns, timeout=timeout)
            except pexpect.EOF as eof_except:
              self.child.close()
              raise LLDBPExpectException('EOF', patterns, self.child)
            except pexpect.TIMEOUT as timeout_except:
              raise LLDBPExpectException('TIMEOUT', patterns, self.child)

        def expectall(self, patterns=None, timeout=None, exact=None):
            if patterns is None: return None
            if timeout is None: timeout = self.timeout
            if exact is None: exact = False
            for pattern in patterns:
                self.expect(pattern, timeout=timeout, exact=exact)

        def sendimpl(self, sender, command, patterns=None, timeout=None, exact=None):
            sender(command)
            return self.expect(patterns=patterns, timeout=timeout, exact=exact)

        def send(self, command, patterns=None, timeout=None, exact=None):
            return self.sendimpl(self.child.send, command, patterns, timeout, exact)

        def sendline(self, command, patterns=None, timeout=None, exact=None):
            return self.sendimpl(self.child.sendline, command, patterns, timeout, exact)

        def quit(self, gracefully=None):
            if gracefully is None: gracefully = True
            self.child.sendeof()
            self.child.close(force=not gracefully)
            self.child = None

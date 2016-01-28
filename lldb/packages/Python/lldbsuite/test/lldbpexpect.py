from __future__ import print_function
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

if sys.platform.startswith('win32'):
    class PExpectTest(TestBase):
        pass
else:
    import pexpect

    class PExpectTest(TestBase):
    
        mydir = TestBase.compute_mydir(__file__)

        def setUp(self):
            TestBase.setUp(self)

        def launchArgs(self):
            pass

        def launch(self, timeout=None):
            if timeout is None: timeout = 30
            logfile = sys.stdout if self.TraceOn() else None
            self.child = pexpect.spawn('%s %s' % (lldbtest_config.lldbExec, self.launchArgs()), logfile=logfile)
            self.child.timeout = timeout
            self.timeout = timeout

        def expect(self, patterns=None, timeout=None, exact=None):
            if patterns is None: return None
            if timeout is None: timeout = self.timeout
            if exact is None: exact = False
            if exact:
                return self.child.expect_exact(patterns, timeout=timeout)
            else:
                return self.child.expect(patterns, timeout=timeout)

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

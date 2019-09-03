# lldbrepl.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
import lldb
from lldbsuite.test.lldbpexpect import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2
import sys
import pexpect
import swift
import re
from lldbsuite.test.decorators import no_debug_info_test, skipIfLinux
from lldbsuite.test.decorators import swiftTest


class REPLTest(TestBase):

    PROMPT = re.compile('\\d+>')

    def expect_prompt(self):
        self.child.expect(self.PROMPT, timeout=30)

    def launch(self, timeout=30):
        logfile = getattr(sys.stdout, 'buffer',
                          sys.stdout) if self.TraceOn() else None
        args = [
            '--no-lldbinit', '--no-use-colors', '-x',
            '--repl=-enable-objc-interop -sdk %s'.format(
                swift.getSwiftSDKRoot())
        ]
        self.child = pexpect.spawn(
            lldbtest_config.lldbExec,
            args=args,
            logfile=logfile,
            timeout=timeout)
        self.child.expect('Welcome to.*Swift')
        self.child.expect_exact('Type :help for assistance')
        self.expect_prompt()

    @swiftTest
    @no_debug_info_test
    def testREPL(self):
        try:
            self.launch(timeout=30)
            self.doTest()
        finally:
            try:
                # and quit
                self.quit(gracefully=False)
            except:
                pass

    def command(self,
                cmd,
                patterns=None,
                timeout=30,
                exact=False,
                prompt_sync=True):
        self.child.sendline(cmd)
        if patterns is not None:
            for pattern in patterns:
                if exact:
                    self.child.expect_exact(patterns, timeout=timeout)
                else:
                    self.child.expect(patterns, timeout=timeout)
        if prompt_sync:
            self.expect_prompt()


def load_tests(x, y, z):
    tests = []
    for test in y:
        testcase = test._tests[0]
        if isinstance(testcase, REPLTest):
            continue
        tests.append(test)
    return tests

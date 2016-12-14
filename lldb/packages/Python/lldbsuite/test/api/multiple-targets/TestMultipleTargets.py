"""Test the lldb public C++ api when creating multiple targets simultaneously."""

from __future__ import print_function


import os
import re
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestMultipleTargets(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfNoSBHeaders
    @skipIfHostIncompatibleWithRemote
    @expectedFailureAll(
        oslist=["windows", "freebsd"],
        bugnumber="llvm.org/pr20282")
    def test_multiple_targets(self):
        env = {self.dylibPath: self.getLLDBLibraryEnvVal()}

        self.driver_exe = os.path.join(os.getcwd(), "multi-target")
        self.buildDriver('main.cpp', self.driver_exe)
        self.addTearDownHook(lambda: os.remove(self.driver_exe))
        self.signBinary(self.driver_exe)

# check_call will raise a CalledProcessError if multi-process-driver doesn't return
# exit code 0 to indicate success.  We can let this exception go - the test harness
# will recognize it as a test failure.

        if self.TraceOn():
            print("Running test %s" % self.driver_exe)
            check_call([self.driver_exe, self.driver_exe], env=env)
        else:
            with open(os.devnull, 'w') as fnull:
                check_call([self.driver_exe, self.driver_exe],
                           env=env, stdout=fnull, stderr=fnull)

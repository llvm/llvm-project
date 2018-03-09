"""
Test the output of `frame diagnose` for an array access
"""

from __future__ import print_function

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestArray(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    @skipUnlessDarwin
    @skipIfDarwinEmbedded  # <rdar://problem/33842388> frame diagnose doesn't work for armv7 or arm64
    def test_array(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        self.runCmd("run", RUN_SUCCEEDED)
        self.expect("thread list", "Thread should be stopped",
                    substrs=['stopped'])
        self.expect(
            "frame diagnose",
            "Crash diagnosis was accurate",
            substrs=["a[10]"])

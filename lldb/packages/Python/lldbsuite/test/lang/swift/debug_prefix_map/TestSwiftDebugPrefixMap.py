# TestSwiftDebugPrefixMap.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2018 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Test that LLDB correctly finds source when debug info is remapped.
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import shutil
import unittest2


class TestSwiftDebugPrefixMap(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.add_test_categories(['swiftpr'])
    def test_debug_prefix_map(self):
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)

    def do_test(self):
        cwd = os.path.dirname(os.path.realpath(__file__))

        self.build()

        # Mirror the same source tree layout used in the Makefile. When lldb is
        # invoked in the CWD, it should find the source files with the same
        # relative paths used during compilation because the compiler's CWD was
        # remapped to ".".
        src = os.path.join(cwd, 'main.swift')
        local_srcroot = os.path.join(cwd, 'srcroot')
        local_main = os.path.join(local_srcroot, 'main.swift')

        if not os.path.exists(local_srcroot):
            os.makedirs(local_srcroot)
        shutil.copy(src, local_main)

        # Clean up the files we created above when the test ends.
        def _cleanup():
            shutil.rmtree(local_srcroot)
        self.addTearDownHook(_cleanup)

        exe_name = 'a.out'
        exe = self.getBuildArtifact(exe_name)

        # Create the target.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Don't allow ANSI highlighting to interfere with the output.
        self.runCmd('settings set stop-show-column none')
        self.expect('breakpoint set -l 13', substrs=['foo'])
        self.expect('source list -l 13', substrs=['return x + y - z'])
        self.expect('run', substrs=['return x + y - z'])

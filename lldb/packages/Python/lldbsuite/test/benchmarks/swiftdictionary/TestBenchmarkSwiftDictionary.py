# coding=utf-8

# TestBenchmarkSwiftDictionary.py
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

"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.lldbbench import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil


class TestBenchmarkSwiftDictionary(BenchBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.benchmarks_test
    def test_run_command(self):
        """Benchmark the Swift dictionary data formatter"""
        self.build()
        self.data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        BenchBase.setUp(self)

    def data_formatter_commands(self):
        """Benchmark the Swift dictionary data formatter"""
        self.runCmd("file " + getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        bkpt = self.target().FindBreakpointByID(
            lldbutil.run_break_set_by_source_regexp(
                self, "break here"))

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)
            self.runCmd(
                "settings set target.max-children-count 256",
                check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        sw = Stopwatch()

        sw.start()
        self.expect('frame variable -A dict', substrs=['[300]', '300'])
        sw.stop()

        print("time to print: %s" % (sw))

"""Test that the Objective-C syntax for dictionary/array literals and indexing works"""

from __future__ import print_function


import unittest2
import os
import time
import platform

from distutils.version import StrictVersion

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCNewSyntaxTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.m', '// Set breakpoint 0 here.')

    @skipUnlessDarwin
    @expectedFailureAll(
        oslist=['macosx'],
        compiler='clang',
        compiler_version=[
            '<',
            '7.0.0'])
    @expectedFailureAll(
        oslist=['macosx'],
        debug_info="gmodules",
        bugnumber="rdar://26558099")
    @unittest2.skipIf(platform.system() != "Darwin" or StrictVersion(
        '12.0.0') > platform.release(), "Only supported on Darwin 12.0.0+")
    def test_expr(self):
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        self.expect(
            "expr --object-description -- immutable_array[0]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["foo"])

        self.expect(
            "expr --object-description -- mutable_array[0]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["foo"])

        self.expect(
            "expr --object-description -- mutable_array[0] = @\"bar\"",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["bar"])

        self.expect(
            "expr --object-description -- mutable_array[0]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["bar"])

        self.expect(
            "expr --object-description -- immutable_dictionary[@\"key\"]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["value"])

        self.expect(
            "expr --object-description -- mutable_dictionary[@\"key\"]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["value"])

        self.expect(
            "expr --object-description -- mutable_dictionary[@\"key\"] = @\"object\"",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["object"])

        self.expect(
            "expr --object-description -- mutable_dictionary[@\"key\"]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["object"])

        self.expect(
            "expr --object-description -- @[ @\"foo\", @\"bar\" ]",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "NSArray",
                "foo",
                "bar"])

        self.expect(
            "expr --object-description -- @{ @\"key\" : @\"object\" }",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "key",
                "object"])

        self.expect("expr --object-description -- @'a'",
                    VARIABLES_DISPLAYED_CORRECTLY, substrs=[str(ord('a'))])

        self.expect(
            "expr --object-description -- @1",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["1"])

        self.expect(
            "expr --object-description -- @1l",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["1"])

        self.expect(
            "expr --object-description -- @1ul",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["1"])

        self.expect(
            "expr --object-description -- @1ll",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["1"])

        self.expect(
            "expr --object-description -- @1ull",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["1"])

        self.expect("expr -- @123.45", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["NSNumber", "123.45"])

        self.expect(
            "expr --object-description -- @( 1 + 3 )",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["4"])
        self.expect(
            "expr -- @((char*)\"Hello world\" + 6)",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "NSString",
                "world"])

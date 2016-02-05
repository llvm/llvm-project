"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function



import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class LibcxxUnorderedDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows # libc++ not ported to Windows yet
    @skipIfGcc
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_source_regexp (self, "Set break point at this line.")

        self.runCmd("run", RUN_SUCCEEDED)

        lldbutil.skip_if_library_missing(self, self.target(), lldbutil.PrintableRegex("libc\+\+"))

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)
            self.runCmd("settings set target.max-children-count 256", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.expect('image list', substrs = self.getLibcPlusPlusLibs())

        self.look_for_content_and_continue(
            "map", ['size=5 {', 'hello', 'world', 'this', 'is', 'me'])

        self.look_for_content_and_continue(
            "mmap", ['size=6 {', 'first = 3', 'second = "this"',
                                 'first = 2', 'second = "hello"'])

        self.look_for_content_and_continue(
            "iset", ['size=5 {', '\[\d\] = 5', '\[\d\] = 3', '\[\d\] = 2'])

        self.look_for_content_and_continue(
            "sset", ['size=5 {', '\[\d\] = "is"', '\[\d\] = "world"',
                                 '\[\d\] = "hello"'])

        self.look_for_content_and_continue(
            "imset", ['size=6 {', '(\[\d\] = 3(\\n|.)+){3}',
                                  '\[\d\] = 2', '\[\d\] = 1'])

        self.look_for_content_and_continue(
            "smset",
            ['size=5 {', '(\[\d\] = "is"(\\n|.)+){2}',
                         '(\[\d\] = "world"(\\n|.)+){2}'])

    def look_for_content_and_continue(self, var_name, patterns):
        self.expect( ("frame variable %s" % var_name), patterns=patterns)
        self.runCmd("continue")

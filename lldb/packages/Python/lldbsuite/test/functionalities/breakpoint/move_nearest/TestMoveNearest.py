from __future__ import print_function


import unittest2
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestMoveNearest(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line1 = line_number('foo.h', '// !BR1')
        self.line2 = line_number('foo.h', '// !BR2')
        self.line_between = line_number('main.cpp', "// BR_Between")
        print("BR_Between found at", self.line_between)
        self.line_main = line_number('main.cpp', '// !BR_main')

    def test(self):
        """Test target.move-to-nearest logic"""

        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        lldbutil.run_break_set_by_symbol(self, 'main', sym_exact=True)
        environment = self.registerSharedLibrariesWithTarget(target, ["foo"])
        process = target.LaunchSimple(None, environment, self.get_process_working_directory())
        self.assertEquals(process.GetState(), lldb.eStateStopped)

        # Regardless of the -m value the breakpoint should have exactly one
        # location on the foo functions
        self.runCmd("settings set target.move-to-nearest-code true")
        lldbutil.run_break_set_by_file_and_line(self, 'foo.h', self.line1,
                loc_exact=True, extra_options="-m 1")
        lldbutil.run_break_set_by_file_and_line(self, 'foo.h', self.line2,
                loc_exact=True, extra_options="-m 1")

        self.runCmd("settings set target.move-to-nearest-code false")
        lldbutil.run_break_set_by_file_and_line(self, 'foo.h', self.line1,
                loc_exact=True, extra_options="-m 0")
        lldbutil.run_break_set_by_file_and_line(self, 'foo.h', self.line2,
                loc_exact=True, extra_options="-m 0")


        # Make sure we set a breakpoint in main with -m 1 for various lines in
        # the function declaration
        # "int"
        lldbutil.run_break_set_by_file_and_line(self, 'main.cpp',
                self.line_main-1, extra_options="-m 1")
        # "main()"
        lldbutil.run_break_set_by_file_and_line(self, 'main.cpp',
                self.line_main, extra_options="-m 1")
        # "{"
        lldbutil.run_break_set_by_file_and_line(self, 'main.cpp',
                self.line_main+1, extra_options="-m 1")
        # "return .."
        lldbutil.run_break_set_by_file_and_line(self, 'main.cpp',
                self.line_main+2, extra_options="-m 1")

        # Make sure we don't put move the breakpoint if it is set between two functions:
        lldbutil.run_break_set_by_file_and_line(self, 'main.cpp',
                self.line_between, extra_options="-m 1", num_expected_locations=0)

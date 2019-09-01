"""
Test that the po command acts correctly.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class PoVerbosityTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.line = line_number('main.m',
                                '// Stop here')

    @skipUnlessDarwin
    def test(self):
        """Test that the po command acts correctly."""
        self.build()

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synthetic clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        """Test expr + formatters for good interoperability."""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("expr -O -v -- foo",
                    substrs=['(id) $', ' = 0x', '1 = 2', '2 = 3;'])
        self.expect("expr -O -vfull -- foo",
                    substrs=['(id) $', ' = 0x', '1 = 2', '2 = 3;'])
        self.expect("expr -O -- foo", matching=False,
                    substrs=['(id) $'])

        self.expect("expr -O -- 22", matching=False,
                    substrs=['(int) $'])
        self.expect("expr -O -- 22",
                    substrs=['22'])

        self.expect("expr -O -vfull -- 22",
                    substrs=['(int) $', ' = 22'])

        self.expect("expr -O -v -- 22",
                    substrs=['(int) $', ' = 22'])

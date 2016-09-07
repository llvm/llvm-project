"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class NamedSummariesDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

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

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd(
            "type summary add --summary-string \"AllUseIt: x=${var.x} {y=${var.y}} {z=${var.z}}\" --name AllUseIt")
        self.runCmd(
            "type summary add --summary-string \"First: x=${var.x} y=${var.y} dummy=${var.dummy}\" First")
        self.runCmd(
            "type summary add --summary-string \"Second: x=${var.x} y=${var.y%hex}\" Second")
        self.runCmd(
            "type summary add --summary-string \"Third: x=${var.x} z=${var.z}\" Third")

        self.expect('type summary list', substrs=['AllUseIt'])

        self.expect("frame variable first",
                    substrs=['First: x=12'])

        self.expect("frame variable first --summary AllUseIt",
                    substrs=['AllUseIt: x=12'])

        # We *DO NOT* remember the summary choice anymore
        self.expect("frame variable first", matching=False,
                    substrs=['AllUseIt: x=12'])
        self.expect("frame variable first",
                    substrs=['First: x=12'])

        self.runCmd("thread step-over")  # 2

        self.expect("frame variable first",
                    substrs=['First: x=12'])

        self.expect("frame variable first --summary AllUseIt",
                    substrs=['AllUseIt: x=12',
                             'y=34'])

        self.expect("frame variable second --summary AllUseIt",
                    substrs=['AllUseIt: x=65',
                             'y=43.25'])

        self.expect("frame variable third --summary AllUseIt",
                    substrs=['AllUseIt: x=96',
                             'z=',
                             'E'])

        self.runCmd("thread step-over")  # 3

        self.expect("frame variable second",
                    substrs=['Second: x=65',
                             'y=0x'])

        # <rdar://problem/11576143> decided that invalid summaries will raise an error
        # instead of just defaulting to the base summary
        self.expect(
            "frame variable second --summary NoSuchSummary",
            error=True,
            substrs=['must specify a valid named summary'])

        self.runCmd("thread step-over")

        self.runCmd(
            "type summary add --summary-string \"FirstAndFriends: x=${var.x} {y=${var.y}} {z=${var.z}}\" First --name FirstAndFriends")

        self.expect("frame variable first",
                    substrs=['FirstAndFriends: x=12',
                             'y=34'])

        self.runCmd("type summary delete First")

        self.expect("frame variable first --summary FirstAndFriends",
                    substrs=['FirstAndFriends: x=12',
                             'y=34'])

        self.expect("frame variable first", matching=True,
                    substrs=['x = 12',
                             'y = 34'])

        self.runCmd("type summary delete FirstAndFriends")
        self.expect("type summary delete NoSuchSummary", error=True)
        self.runCmd("type summary delete AllUseIt")

        self.expect("frame variable first", matching=False,
                    substrs=['FirstAndFriends'])

        self.runCmd("thread step-over")  # 4

        self.expect("frame variable first", matching=False,
                    substrs=['FirstAndFriends: x=12',
                             'y=34'])

"""Look up enum type information and check for correct display."""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *


class EnumTypesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    # derefing the null pointer "works" on Windows
    @expectedFailureAll(oslist=['windows'])
    def test(self):
        """Test 'image lookup -t days' and check for correct display and enum value printing."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        bkpt_id = lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # Look up information about the 'days' enum type.
        # Check for correct display.
        self.expect("image lookup -t days", DATA_TYPES_DISPLAYED_CORRECTLY,
                    substrs=['enum days {',
                             'Monday',
                             'Tuesday',
                             'Wednesday',
                             'Thursday',
                             'Friday',
                             'Saturday',
                             'Sunday',
                             'kNumDays',
                             '}'])

        enum_values = ['-4',
                       'Monday',
                       'Tuesday',
                       'Wednesday',
                       'Thursday',
                       'Friday',
                       'Saturday',
                       'Sunday',
                       'kNumDays',
                       '5']

        # Make sure a pointer to an anonymous enum type does crash LLDB and displays correctly using
        # frame variable and expression commands
        self.expect(
            'frame variable f.op',
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=[
                'ops *',
                'f.op'],
            patterns=['0x0+$'])
        self.expect(
            'frame variable *f.op',
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=[
                'ops',
                '*f.op',
                '<parent is NULL>'])
        self.expect(
            'expr f.op',
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=[
                'ops *',
                '$'],
            patterns=['0x0+$'])
        self.expect(
            'expr *f.op',
            DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs=['error:'],
            error=True)

        bkpt = self.target().FindBreakpointByID(bkpt_id)
        for enum_value in enum_values:
            self.expect(
                "frame variable day",
                'check for valid enumeration value',
                substrs=[enum_value])
            lldbutil.continue_to_breakpoint(self.process(), bkpt)

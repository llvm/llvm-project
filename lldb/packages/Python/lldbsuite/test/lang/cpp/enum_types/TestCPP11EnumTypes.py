"""Look up enum type information and check for correct display."""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class CPP11EnumTypesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_int8_t(self):
        """Test C++11 enumeration class types as int8_t types."""
        self.build(
            dictionary={
                'CFLAGS_EXTRAS': '"-DTEST_BLOCK_CAPTURED_VARS=int8_t"'})
        self.image_lookup_for_enum_type()

    def test_int16_t(self):
        """Test C++11 enumeration class types as int16_t types."""
        self.build(
            dictionary={
                'CFLAGS_EXTRAS': '"-DTEST_BLOCK_CAPTURED_VARS=int16_t"'})
        self.image_lookup_for_enum_type()

    def test_int32_t(self):
        """Test C++11 enumeration class types as int32_t types."""
        self.build(
            dictionary={
                'CFLAGS_EXTRAS': '"-DTEST_BLOCK_CAPTURED_VARS=int32_t"'})
        self.image_lookup_for_enum_type()

    def test_int64_t(self):
        """Test C++11 enumeration class types as int64_t types."""
        self.build(
            dictionary={
                'CFLAGS_EXTRAS': '"-DTEST_BLOCK_CAPTURED_VARS=int64_t"'})
        self.image_lookup_for_enum_type()

    def test_uint8_t(self):
        """Test C++11 enumeration class types as uint8_t types."""
        self.build(
            dictionary={
                'CFLAGS_EXTRAS': '"-DTEST_BLOCK_CAPTURED_VARS=uint8_t"'})
        self.image_lookup_for_enum_type()

    def test_uint16_t(self):
        """Test C++11 enumeration class types as uint16_t types."""
        self.build(
            dictionary={
                'CFLAGS_EXTRAS': '"-DTEST_BLOCK_CAPTURED_VARS=uint16_t"'})
        self.image_lookup_for_enum_type()

    def test_uint32_t(self):
        """Test C++11 enumeration class types as uint32_t types."""
        self.build(
            dictionary={
                'CFLAGS_EXTRAS': '"-DTEST_BLOCK_CAPTURED_VARS=uint32_t"'})
        self.image_lookup_for_enum_type()

    def test_uint64_t(self):
        """Test C++11 enumeration class types as uint64_t types."""
        self.build(
            dictionary={
                'CFLAGS_EXTRAS': '"-DTEST_BLOCK_CAPTURED_VARS=uint64_t"'})
        self.image_lookup_for_enum_type()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def image_lookup_for_enum_type(self):
        """Test C++11 enumeration class types."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        bkpt_id = lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # Look up information about the 'DayType' enum type.
        # Check for correct display.
        self.expect("image lookup -t DayType", DATA_TYPES_DISPLAYED_CORRECTLY,
                    patterns=['enum( struct| class) DayType {'],
                    substrs=['Monday',
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

        bkpt = self.target().FindBreakpointByID(bkpt_id)
        for enum_value in enum_values:
            self.expect(
                "frame variable day",
                'check for valid enumeration value',
                substrs=[enum_value])
            lldbutil.continue_to_breakpoint(self.process(), bkpt)

# coding=utf8
"""
Test that the C++11 support for char16_t and char32_t works correctly.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class Char1632TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.source = 'main.cpp'
        self.lines = [line_number(self.source, '// breakpoint1'),
                      line_number(self.source, '// breakpoint2')]

    @expectedFailureAll(
        compiler="icc",
        bugnumber="ICC (13.1) does not emit the DW_TAG_base_type for char16_t and char32_t.")
    def test(self):
        """Test that the C++11 support for char16_t and char32_t works correctly."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set breakpoints
        for line in self.lines:
            lldbutil.run_break_set_by_file_and_line(self, "main.cpp", line)

        # Now launch the process, and do not stop at entry point and stop at
        # breakpoint1
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        if not process:
            self.fail("SBTarget.Launch() failed")

        if self.TraceOn():
            self.runCmd("frame variable")

        # Check that we correctly report the const types
        self.expect(
            "frame variable cs16 cs32",
            substrs=[
                '(const char16_t *) cs16 = ',
                '(const char32_t *) cs32 = ',
                'u"hello world ྒྙྐ"',
                'U"hello world ྒྙྐ"'])

        # Check that we correctly report the non-const types
        self.expect(
            "frame variable s16 s32",
            substrs=[
                '(char16_t *) s16 = ',
                '(char32_t *) s32 = ',
                'u"ﺸﺵۻ"',
                'U"ЕЙРГЖО"'])

        # Check that we correctly report the array types
        self.expect(
            "frame variable as16 as32",
            patterns=[
                '\(char16_t \[[0-9]+\]\) as16 = ',
                '\(char32_t \[[0-9]+\]\) as32 = '],
            substrs=[
                'u"ﺸﺵۻ"',
                'U"ЕЙРГЖО"'])

        self.runCmd("next")  # step to after the string is nullified

        # check that we don't crash on NULL
        self.expect("frame variable s32",
                    substrs=['(char32_t *) s32 = 0x00000000'])

        # continue and hit breakpoint2
        self.runCmd("continue")

        # check that the new strings show
        self.expect(
            "frame variable s16 s32",
            substrs=[
                '(char16_t *) s16 = 0x',
                '(char32_t *) s32 = ',
                '"色ハ匂ヘト散リヌルヲ"',
                '"෴"'])

        # check the same as above for arrays
        self.expect(
            "frame variable as16 as32",
            patterns=[
                '\(char16_t \[[0-9]+\]\) as16 = ',
                '\(char32_t \[[0-9]+\]\) as32 = '],
            substrs=[
                '"色ハ匂ヘト散リヌルヲ"',
                '"෴"'])

        # check that zero values are properly handles
        self.expect('frame variable cs16_zero', substrs=["U+0000 u'\\0'"])
        self.expect(
            'frame variable cs32_zero',
            substrs=["U+0x00000000 U'\\0'"])
        self.expect('expression cs16_zero', substrs=["U+0000 u'\\0'"])
        self.expect('expression cs32_zero', substrs=["U+0x00000000 U'\\0'"])

        # Check that we can run expressions that return charN_t
        self.expect("expression u'a'", substrs=['(char16_t) $', "61 u'a'"])
        self.expect("expression U'a'", substrs=['(char32_t) $', "61 U'a'"])

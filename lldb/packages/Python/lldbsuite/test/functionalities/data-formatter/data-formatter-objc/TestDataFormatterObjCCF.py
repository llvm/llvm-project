# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from ObjCDataFormatterTestCase import ObjCDataFormatterTestCase


class ObjCDataFormatterCF(ObjCDataFormatterTestCase):

    @skipUnlessDarwin
    def test_coreframeworks_and_run_command(self):
        """Test formatters for Core OSX frameworks."""
        self.build()
        self.target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, '// Set break point at this line.',
            lldb.SBFileSpec('main.m', False))

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=['stopped', 'stop reason = breakpoint'])

        # check formatters for common Objective-C types
        expect_strings = [
            '(CFGregorianUnits) cf_greg_units = 1 years, 3 months, 5 days, 12 hours, 5 minutes 7 seconds',
            '(CFRange) cf_range = location=4 length=4',
            '(NSPoint) ns_point = (x = 4, y = 4)',
            '(NSRange) ns_range = location=4, length=4',
            '(NSRect) ns_rect = (origin = (x = 1, y = 1), size = (width = 5, height = 5))',
            '(NSRectArray) ns_rect_arr = ((x = 1, y = 1), (width = 5, height = 5)), ...',
            '(NSSize) ns_size = (width = 5, height = 7)',
            '(CGSize) cg_size = (width = 1, height = 6)',
            '(CGPoint) cg_point = (x = 2, y = 7)',
            '(CGRect) cg_rect = (origin = (x = 1, y = 2), size = (width = 7, height = 7))',
            '(Rect) rect = (t=4, l=8, b=4, r=7)',
            '(Rect *) rect_ptr = (t=4, l=8, b=4, r=7)',
            '(Point) point = (v=7, h=12)', '(Point *) point_ptr = (v=7, h=12)',
            '1985', 'foo_selector_impl'
        ]

        if self.getArchitecture() in ['i386', 'x86_64']:
            expect_strings.append('(HIPoint) hi_point = (x=7, y=12)')
            expect_strings.append(
                '(HIRect) hi_rect = origin=(x = 3, y = 5) size=(width = 4, height = 6)'
            )
            expect_strings.append(
                '(RGBColor) rgb_color = red=3 green=56 blue=35')
            expect_strings.append(
                '(RGBColor *) rgb_color_ptr = red=3 green=56 blue=35')

        self.expect("frame variable", substrs=expect_strings)

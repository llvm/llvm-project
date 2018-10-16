# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function


import datetime
import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    def test_plain_objc_with_run_command(self):
        """Test basic ObjC formatting behavior."""
        self.build()
        self.plain_data_formatter_commands()

    def appkit_tester_impl(self, commands):
        self.build()
        self.appkit_common_data_formatters_command()
        commands()

    @skipUnlessDarwin
    def test_nsnumber_with_run_command(self):
        """Test formatters for NSNumber."""
        self.appkit_tester_impl(self.nsnumber_data_formatter_commands)

    @skipUnlessDarwin
    def test_nscontainers_with_run_command(self):
        """Test formatters for  NS container classes."""
        self.appkit_tester_impl(self.nscontainers_data_formatter_commands)

    @skipUnlessDarwin
    def test_nsdata_with_run_command(self):
        """Test formatters for  NSData."""
        self.appkit_tester_impl(self.nsdata_data_formatter_commands)

    @skipUnlessDarwin
    def test_nsurl_with_run_command(self):
        """Test formatters for NSURL."""
        self.appkit_tester_impl(self.nsurl_data_formatter_commands)

    @skipUnlessDarwin
    @expectedFailureAll(bugnumber="rdar://25587546")
    def test_nserror_with_run_command(self):
        """Test formatters for NSError."""
        self.appkit_tester_impl(self.nserror_data_formatter_commands)

    @skipUnlessDarwin
    def test_nsbundle_with_run_command(self):
        """Test formatters for NSBundle."""
        self.appkit_tester_impl(self.nsbundle_data_formatter_commands)

    @skipUnlessDarwin
    def test_nsexception_with_run_command(self):
        """Test formatters for NSException."""
        self.appkit_tester_impl(self.nsexception_data_formatter_commands)

    @skipUnlessDarwin
    def test_nsdate_with_run_command(self):
        """Test formatters for NSDate."""
        self.appkit_tester_impl(self.nsdate_data_formatter_commands)

    @skipUnlessDarwin
    def test_coreframeworks_and_run_command(self):
        """Test formatters for Core OSX frameworks."""
        self.build()
        self.cf_data_formatter_commands()

    @skipUnlessDarwin
    def test_kvo_with_run_command(self):
        """Test the behavior of formatters when KVO is in use."""
        self.build()
        self.kvo_data_formatter_commands()

    @skipUnlessDarwin
    def test_expr_with_run_command(self):
        """Test common cases of expression parser <--> formatters interaction."""
        self.build()
        self.expr_objc_data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.m', '// Set break point at this line.')

    def plain_data_formatter_commands(self):
        """Test basic ObjC formatting behavior."""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

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
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("type summary add --summary-string \"${var%@}\" MyClass")

        self.expect("frame variable object2",
                    substrs=['MyOtherClass'])

        self.expect("frame variable *object2",
                    substrs=['MyOtherClass'])

        # Now let's delete the 'MyClass' custom summary.
        self.runCmd("type summary delete MyClass")

        # The type format list should not show 'MyClass' at this point.
        self.expect("type summary list", matching=False,
                    substrs=['MyClass'])

        self.runCmd("type summary add --summary-string \"a test\" MyClass")

        self.expect("frame variable *object2",
                    substrs=['*object2 =',
                             'MyClass = a test',
                             'backup = '])

        self.expect("frame variable object2", matching=False,
                    substrs=['a test'])

        self.expect("frame variable object",
                    substrs=['a test'])

        self.expect("frame variable *object",
                    substrs=['a test'])

        self.expect('frame variable myclass',
                    substrs=['(Class) myclass = NSValue'])
        self.expect('frame variable myclass2',
                    substrs=['(Class) myclass2 = ', 'NS', 'String'])
        self.expect('frame variable myclass3',
                    substrs=['(Class) myclass3 = Molecule'])
        self.expect('frame variable myclass4',
                    substrs=['(Class) myclass4 = NSMutableArray'])
        self.expect('frame variable myclass5',
                    substrs=['(Class) myclass5 = nil'])

    def appkit_common_data_formatters_command(self):
        """Test formatters for AppKit classes."""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

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
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

    def nsnumber_data_formatter_commands(self):
        # Now enable AppKit and check we are displaying Cocoa classes correctly
        self.expect('frame variable num1 num2 num3 num5 num6 num7 num8_Y num8_N num9',
                    substrs=['(NSNumber *) num1 = ', ' (int)5',
                             '(NSNumber *) num2 = ', ' (float)3.1',
                             '(NSNumber *) num3 = ', ' (double)3.14',
                             '(NSNumber *) num5 = ', ' (char)65',
                             '(NSNumber *) num6 = ', ' (long)255',
                             '(NSNumber *) num7 = ', '2000000',
                             '(NSNumber *) num8_Y = ', 'YES',
                             '(NSNumber *) num8_N = ', 'NO',
                             '(NSNumber *) num9 = ', ' (short)-31616'])


        self.runCmd('frame variable num4', check=True)
        output = self.res.GetOutput()
        i128_handled_correctly = False

        if output.find('long') >= 0:
            i128_handled_correctly = (output.find('(long)-2') >= 0)
        if output.find('int128_t') >= 0:
            i128_handled_correctly = (output.find('(int128_t)18446744073709551614') >= 0) # deliberately broken, should be ..14

        self.assertTrue(i128_handled_correctly, "Expected valid output for int128_t; got " + output)

        self.expect('frame variable num_at1 num_at2 num_at3 num_at4',
                    substrs=['(NSNumber *) num_at1 = ', ' (int)12',
                             '(NSNumber *) num_at2 = ', ' (int)-12',
                             '(NSNumber *) num_at3 = ', ' (double)12.5',
                             '(NSNumber *) num_at4 = ', ' (double)-12.5'])

    def nsdecimalnumber_data_formatter_commands(self):
        self.expect('frame variable decimal_number decimal_neg_number decimal_one decimal_zero decimal_nan',
                    substrs=['(NSDecimalNumber *) decimal_number = ', '123456 x 10^-10',
                             '(NSDecimalNumber *) decimal_neg_number = ', '-123456 x 10^10',
                             '(NSDecimalNumber *) decimal_one = ', '1 x 10^0',
                             '(NSDecimalNumber *) decimal_zero = ', '0',
                             '(NSDecimalNumber *) decimal_nan = ', 'NaN'])

    def nscontainers_data_formatter_commands(self):
        self.expect(
            'frame variable newArray nsDictionary newDictionary nscfDictionary cfDictionaryRef newMutableDictionary cfarray_ref mutable_array_ref',
            substrs=[
                '(NSArray *) newArray = ',
                '@"50 elements"',
                '(NSDictionary *) newDictionary = ',
                ' 12 key/value pairs',
                '(NSDictionary *) newMutableDictionary = ',
                ' 21 key/value pairs',
                '(NSDictionary *) nsDictionary = ',
                ' 2 key/value pairs',
                '(CFDictionaryRef) cfDictionaryRef = ',
                ' 3 key/value pairs',
                '(CFArrayRef) cfarray_ref = ',
                '@"3 elements"',
                '(CFMutableArrayRef) mutable_array_ref = ',
                '@"11 elements"'])

        self.expect('frame variable iset1 iset2 imset',
                    substrs=['4 indexes', '512 indexes', '10 indexes'])

        self.expect(
            'frame variable binheap_ref',
            substrs=[
                '(CFBinaryHeapRef) binheap_ref = ',
                '@"21 items"'])

        self.expect(
            'expression -d run -- (NSArray*)[NSArray new]',
            substrs=['@"0 elements"'])

    def nsdata_data_formatter_commands(self):
        self.expect(
            'frame variable immutableData mutableData data_ref mutable_data_ref mutable_string_ref concreteData concreteMutableData',
            substrs=[
                '(NSData *) immutableData = ',
                ' 4 bytes',
                '(NSData *) mutableData = ',
                ' 14 bytes',
                '(CFDataRef) data_ref = ',
                '@"5 bytes"',
                '(CFMutableDataRef) mutable_data_ref = ',
                '@"5 bytes"',
                '(CFMutableStringRef) mutable_string_ref = ',
                ' @"Wish ya knew"',
                '(NSData *) concreteData = ',
                ' 100000 bytes',
                '(NSMutableData *) concreteMutableData = ',
                ' 100000 bytes'])


    def nsurl_data_formatter_commands(self):
        self.expect(
            'frame variable cfurl_ref cfchildurl_ref cfgchildurl_ref',
            substrs=[
                '(CFURLRef) cfurl_ref = ',
                '@"http://www.foo.bar',
                'cfchildurl_ref = ',
                '@"page.html -- http://www.foo.bar',
                '(CFURLRef) cfgchildurl_ref = ',
                '@"?whatever -- http://www.foo.bar/page.html"'])

        self.expect(
            'frame variable nsurl nsurl2 nsurl3',
            substrs=[
                '(NSURL *) nsurl = ',
                '@"http://www.foo.bar',
                '(NSURL *) nsurl2 =',
                '@"page.html -- http://www.foo.bar',
                '(NSURL *) nsurl3 = ',
                '@"?whatever -- http://www.foo.bar/page.html"'])

    def nserror_data_formatter_commands(self):
        self.expect('frame variable nserror',
                    substrs=['domain: @"Foobar" - code: 12'])

        self.expect('frame variable  -d run-target nserror->_userInfo',
                    substrs=['2 key/value pairs'])

        self.expect('frame variable nserror->_userInfo',
                    substrs=['2 key/value pairs'])

        self.expect(
            'frame variable nserror->_userInfo --ptr-depth 1 -d run-target',
            substrs=[
                '@"a"',
                '@"b"',
                "1",
                "2"])

    def nsbundle_data_formatter_commands(self):
        self.expect(
            'frame variable bundle_string bundle_url main_bundle',
            substrs=[
                '(NSBundle *) bundle_string = ',
                ' @"/System/Library/Frameworks/Accelerate.framework"',
                '(NSBundle *) bundle_url = ',
                ' @"/System/Library/Frameworks/Foundation.framework"',
                '(NSBundle *) main_bundle = ',
                'data-formatter-objc'])

    def nsexception_data_formatter_commands(self):
        self.expect(
            'frame variable except0 except1 except2 except3',
            substrs=[
                '(NSException *) except0 = ',
                'name: @"TheGuyWhoHasNoName" - reason: @"cuz it\'s funny"',
                '(NSException *) except1 = ',
                'name: @"TheGuyWhoHasNoName~1" - reason: @"cuz it\'s funny"',
                '(NSException *) except2 = ',
                'name: @"TheGuyWhoHasNoName`2" - reason: @"cuz it\'s funny"',
                '(NSException *) except3 = ',
                'name: @"TheGuyWhoHasNoName/3" - reason: @"cuz it\'s funny"'])

    def nsdate_data_formatter_commands(self):
        self.expect(
            'frame variable date1 date2',
            patterns=[
                '(1985-04-10|1985-04-11)',
                '(2011-01-01|2010-12-31)'])

        # this test might fail if we hit the breakpoint late on December 31st of some given year
        # and midnight comes between hitting the breakpoint and running this line of code
        # hopefully the output will be revealing enough in that case :-)
        now_year = '%s-' % str(datetime.datetime.now().year)

        self.expect('frame variable date3', substrs=[now_year])
        self.expect('frame variable date4', substrs=['1970'])
        self.expect('frame variable date5', substrs=[now_year])

        self.expect('frame variable date1_abs date2_abs',
                    substrs=['1985-04', '2011-01'])

        self.expect('frame variable date3_abs', substrs=[now_year])
        self.expect('frame variable date4_abs', substrs=['1970'])
        self.expect('frame variable date5_abs', substrs=[now_year])

        self.expect('frame variable cupertino home europe',
                    substrs=['@"America/Los_Angeles"',
                             '@"Europe/Rome"',
                             '@"Europe/Paris"'])

        self.expect('frame variable cupertino_ns home_ns europe_ns',
                    substrs=['@"America/Los_Angeles"',
                             '@"Europe/Rome"',
                             '@"Europe/Paris"'])

        self.expect(
            'frame variable mut_bv',
            substrs=[
                '(CFMutableBitVectorRef) mut_bv = ',
                '1110 0110 1011 0000 1101 1010 1000 1111 0011 0101 1101 0001 00'])

    def expr_objc_data_formatter_commands(self):
        """Test common cases of expression parser <--> formatters interaction."""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

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
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # check that the formatters are able to deal safely and correctly
        # with ValueObjects that the expression parser returns
        self.expect(
            'expression ((id)@"Hello for long enough to avoid short string types")',
            matching=False,
            substrs=['Hello for long enough to avoid short string types'])

        self.expect(
            'expression -d run -- ((id)@"Hello for long enough to avoid short string types")',
            substrs=['Hello for long enough to avoid short string types'])

        self.expect('expr -d run -- label1',
                    substrs=['Process Name'])

        self.expect(
            'expr -d run -- @"Hello for long enough to avoid short string types"',
            substrs=['Hello for long enough to avoid short string types'])

        self.expect(
            'expr -d run --object-description -- @"Hello for long enough to avoid short string types"',
            substrs=['Hello for long enough to avoid short string types'])
        self.expect('expr -d run --object-description -- @"Hello"',
                    matching=False, substrs=['@"Hello" Hello'])

    def cf_data_formatter_commands(self):
        """Test formatters for Core OSX frameworks."""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

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
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

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
            '(Point) point = (v=7, h=12)',
            '(Point *) point_ptr = (v=7, h=12)',
            '1985',
            'foo_selector_impl']

        if self.getArchitecture() in ['i386', 'x86_64']:
            expect_strings.append('(HIPoint) hi_point = (x=7, y=12)')
            expect_strings.append(
                '(HIRect) hi_rect = origin=(x = 3, y = 5) size=(width = 4, height = 6)')
            expect_strings.append(
                '(RGBColor) rgb_color = red=3 green=56 blue=35')
            expect_strings.append(
                '(RGBColor *) rgb_color_ptr = red=3 green=56 blue=35')

        self.expect("frame variable",
                    substrs=expect_strings)

    def kvo_data_formatter_commands(self):
        """Test the behavior of formatters when KVO is in use."""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

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
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # as long as KVO is implemented by subclassing, this test should succeed
        # we should be able to dynamically figure out that the KVO implementor class
        # is a subclass of Molecule, and use the appropriate summary for it
        self.runCmd("type summary add -s JustAMoleculeHere Molecule")
        self.expect('frame variable molecule', substrs=['JustAMoleculeHere'])
        self.runCmd("next")
        self.expect("thread list",
                    substrs=['stopped',
                             'step over'])
        self.expect('frame variable molecule', substrs=['JustAMoleculeHere'])

        self.runCmd("next")
        # check that NSMutableDictionary's formatter is not confused when
        # dealing with a KVO'd dictionary
        self.expect(
            'frame variable newMutableDictionary',
            substrs=[
                '(NSDictionary *) newMutableDictionary = ',
                ' 21 key/value pairs'])

        lldbutil.run_break_set_by_regexp(self, 'setAtoms')

        self.runCmd("continue")
        self.expect("frame variable _cmd", substrs=['setAtoms:'])

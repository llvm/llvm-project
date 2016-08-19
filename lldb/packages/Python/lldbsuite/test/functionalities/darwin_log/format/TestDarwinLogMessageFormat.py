"""
Test DarwinLog log message formatting options provided by the
StructuredDataDarwinLog plugin.

These tests are currently only supported when running against Darwin
targets.
"""

from __future__ import print_function

import lldb
import re

from lldbsuite.test import decorators
from lldbsuite.test import lldbtest
from lldbsuite.test import darwin_log


class TestDarwinLogMessageFormat(darwin_log.DarwinLogTestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        super(TestDarwinLogMessageFormat, self).setUp()

        # Source filename.
        self.source = 'main.c'

        # Output filename.
        self.exe_name = 'a.out'
        self.d = {'C_SOURCES': self.source, 'EXE': self.exe_name}

        # Locate breakpoint.
        self.line = lldbtest.line_number(self.source, '// break here')

    def tearDown(self):
        # Shut down the process if it's still running.
        if self.child:
            self.runCmd('process kill')
            self.expect_prompt()
            self.runCmd('quit')

        # Let parent clean up
        super(TestDarwinLogMessageFormat, self).tearDown()

    # ==========================================================================
    # Test settings around log message formatting
    # ==========================================================================

    REGEXES = [
        re.compile(r"\[([^]]+)\] This is the log message."),  # Match log
                                                              # with header.
        re.compile(r"This is the log message."),  # Match no-header content.
        re.compile(r"exited with status")         # Fallback if no log emitted.
    ]

    @decorators.skipUnlessDarwin
    def test_display_without_header_works(self):
        """Test that turning off log message headers works as advertised."""
        self.do_test([], expect_regexes=self.REGEXES)

        # We should not match the first pattern as we shouldn't have header
        # content.
        self.assertIsNotNone(self.child.match)
        self.assertFalse((len(self.child.match.groups()) > 0) and
                        (self.child.match.group(1) != ""),
                        "we should not have seen a header")


    @decorators.skipUnlessDarwin
    def test_display_with_header_works(self):
        """Test that displaying any header works."""
        self.do_test(
            ["--timestamp-relative", "--subsystem", "--category",
             "--activity-chain"],
            expect_regexes=self.REGEXES,
            settings_commands=[
                "display-header true"
            ])

        # We should match the first pattern as we should have header
        # content.
        self.assertIsNotNone(self.child.match)
        self.assertTrue((len(self.child.match.groups()) > 0) and
                         (self.child.match.group(1) != ""),
                         "we should have printed a header")

    def assert_header_contains_timestamp(self, header):
        fields = header.split(',')
        self.assertGreater(len(fields), 0,
                           "there should have been header content present")
        self.assertRegexpMatches(fields[0],
                                 r"^\d+:\d{2}:\d{2}.\d{9}$",
                                 "time field should match expected format")

    @decorators.skipUnlessDarwin
    def test_header_timefield_only_works(self):
        """Test that displaying a header with only the timestamp works."""
        self.do_test(["--timestamp-relative"], expect_regexes=self.REGEXES)

        # We should match the first pattern as we should have header
        # content.
        self.assertIsNotNone(self.child.match)
        self.assertTrue((len(self.child.match.groups()) > 0) and
                        (self.child.match.group(1) != ""),
                        "we should have printed a header")
        header = self.child.match.group(1)
        self.assertEqual(len(header.split(',')), 1,
                         "there should only be one header field")
        self.assert_header_contains_timestamp(header)

    @decorators.skipUnlessDarwin
    def test_header_subsystem_only_works(self):
        """Test that displaying a header with only the subsystem works."""
        self.do_test(["--subsystem"], expect_regexes=self.REGEXES)

        # We should match the first pattern as we should have header
        # content.
        self.assertIsNotNone(self.child.match)
        self.assertTrue((len(self.child.match.groups()) > 0) and
                        (self.child.match.group(1) != ""),
                        "we should have printed a header")
        header = self.child.match.group(1)
        self.assertEqual(len(header.split(',')), 1,
                         "there should only be one header field")
        self.assertEquals(header,
                          "subsystem=org.llvm.lldb.test.sub1")

    @decorators.skipUnlessDarwin
    def test_header_category_only_works(self):
        """Test that displaying a header with only the category works."""
        self.do_test(["--category"], expect_regexes=self.REGEXES)

        # We should match the first pattern as we should have header
        # content.
        self.assertIsNotNone(self.child.match)
        self.assertTrue((len(self.child.match.groups()) > 0) and
                        (self.child.match.group(1) != ""),
                        "we should have printed a header")
        header = self.child.match.group(1)
        self.assertEqual(len(header.split(',')), 1,
                         "there should only be one header field")
        self.assertEquals(header,
                          "category=cat1")

    @decorators.skipUnlessDarwin
    def test_header_activity_chain_only_works(self):
        """Test that displaying a header with only the activity chain works."""
        self.do_test(["--activity-chain"], expect_regexes=self.REGEXES)

        # We should match the first pattern as we should have header
        # content.
        self.assertIsNotNone(self.child.match)
        self.assertTrue((len(self.child.match.groups()) > 0) and
                        (self.child.match.group(1) != ""),
                        "we should have printed a header")
        header = self.child.match.group(1)
        self.assertEqual(len(header.split(',')), 1,
                         "there should only be one header field")
        self.assertEquals(header,
                          "activity-chain=parent-activity:child-activity")

    # @decorators.skipUnlessDarwin
    # def test_header_activity_no_chain_only_works(self):
    #     """Test that displaying a header with only the activity works."""
    #     self.do_test(
    #         [],
    #         expect_regexes=self.REGEXES,
    #         settings_commands=[
    #             "display-header true",
    #             "format-include-timestamp false",
    #             "format-include-activity true",
    #             "format-include-category false",
    #             "format-include-subsystem false",
    #             "display-activity-chain false"
    #         ])

    #     # We should match the first pattern as we should have header
    #     # content.
    #     self.assertIsNotNone(self.child.match)
    #     self.assertTrue((len(self.child.match.groups()) > 0) and
    #                     (self.child.match.group(1) != ""),
    #                     "we should have printed a header")
    #     header = self.child.match.group(1)
    #     self.assertEqual(len(header.split(',')), 1,
    #                      "there should only be one header field")
    #     self.assertEquals(header,
    #                       "activity=child-activity")

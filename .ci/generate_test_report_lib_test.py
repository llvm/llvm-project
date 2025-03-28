# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from io import StringIO
from textwrap import dedent

from junitparser import JUnitXml

import generate_test_report_lib


def junit_from_xml(xml):
    return JUnitXml.fromfile(StringIO(xml))


class TestReports(unittest.TestCase):
    def test_title_only(self):
        self.assertEqual(
            generate_test_report_lib.generate_report("Foo", 0, []), ("", "success")
        )

    def test_no_tests_in_testsuite(self):
        self.assertEqual(
            generate_test_report_lib.generate_report(
                "Foo",
                1,
                [
                    junit_from_xml(
                        dedent(
                            """\
          <?xml version="1.0" encoding="UTF-8"?>
          <testsuites time="0.00">
          <testsuite name="Empty" tests="0" failures="0" skipped="0" time="0.00">
          </testsuite>
          </testsuites>"""
                        )
                    )
                ],
            ),
            ("", None),
        )

    def test_no_failures(self):
        self.assertEqual(
            generate_test_report_lib.generate_report(
                "Foo",
                0,
                [
                    junit_from_xml(
                        dedent(
                            """\
          <?xml version="1.0" encoding="UTF-8"?>
          <testsuites time="0.00">
          <testsuite name="Passed" tests="1" failures="0" skipped="0" time="0.00">
          <testcase classname="Bar/test_1" name="test_1" time="0.00"/>
          </testsuite>
          </testsuites>"""
                        )
                    )
                ],
            ),
            (
                dedent(
                    """\
              # Foo

              * 1 test passed"""
                ),
                "success",
            ),
        )

    def test_no_failures_build_failed(self):
        self.assertEqual(
            generate_test_report_lib.generate_report(
                "Foo",
                1,
                [
                    junit_from_xml(
                        dedent(
                            """\
          <?xml version="1.0" encoding="UTF-8"?>
          <testsuites time="0.00">
          <testsuite name="Passed" tests="1" failures="0" skipped="0" time="0.00">
          <testcase classname="Bar/test_1" name="test_1" time="0.00"/>
          </testsuite>
          </testsuites>"""
                        )
                    )
                ],
                buildkite_info={
                    "BUILDKITE_ORGANIZATION_SLUG": "organization_slug",
                    "BUILDKITE_PIPELINE_SLUG": "pipeline_slug",
                    "BUILDKITE_BUILD_NUMBER": "build_number",
                    "BUILDKITE_JOB_ID": "job_id",
                },
            ),
            (
                dedent(
                    """\
              # Foo

              * 1 test passed

              All tests passed but another part of the build **failed**.

              [Download](https://buildkite.com/organizations/organization_slug/pipelines/pipeline_slug/builds/build_number/jobs/job_id/download.txt) the build's log file to see the details."""
                ),
                "error",
            ),
        )

    def test_report_single_file_single_testsuite(self):
        self.assertEqual(
            generate_test_report_lib.generate_report(
                "Foo",
                1,
                [
                    junit_from_xml(
                        dedent(
                            """\
          <?xml version="1.0" encoding="UTF-8"?>
          <testsuites time="8.89">
          <testsuite name="Bar" tests="4" failures="2" skipped="1" time="410.63">
          <testcase classname="Bar/test_1" name="test_1" time="0.02"/>
          <testcase classname="Bar/test_2" name="test_2" time="0.02">
            <skipped message="Reason"/>
          </testcase>
          <testcase classname="Bar/test_3" name="test_3" time="0.02">
            <failure><![CDATA[Output goes here]]></failure>
          </testcase>
          <testcase classname="Bar/test_4" name="test_4" time="0.02">
            <failure><![CDATA[Other output goes here]]></failure>
          </testcase>
          </testsuite>
          </testsuites>"""
                        )
                    )
                ],
            ),
            (
                dedent(
                    """\
          # Foo

          * 1 test passed
          * 1 test skipped
          * 2 tests failed

          ## Failed Tests
          (click to see output)

          ### Bar
          <details>
          <summary>Bar/test_3/test_3</summary>

          ```
          Output goes here
          ```
          </details>
          <details>
          <summary>Bar/test_4/test_4</summary>

          ```
          Other output goes here
          ```
          </details>"""
                ),
                "error",
            ),
        )

    MULTI_SUITE_OUTPUT = (
        dedent(
            """\
        # ABC and DEF

        * 1 test passed
        * 1 test skipped
        * 2 tests failed

        ## Failed Tests
        (click to see output)

        ### ABC
        <details>
        <summary>ABC/test_2/test_2</summary>

        ```
        ABC/test_2 output goes here
        ```
        </details>

        ### DEF
        <details>
        <summary>DEF/test_2/test_2</summary>

        ```
        DEF/test_2 output goes here
        ```
        </details>"""
        ),
        "error",
    )

    def test_report_single_file_multiple_testsuites(self):
        self.assertEqual(
            generate_test_report_lib.generate_report(
                "ABC and DEF",
                1,
                [
                    junit_from_xml(
                        dedent(
                            """\
          <?xml version="1.0" encoding="UTF-8"?>
          <testsuites time="8.89">
          <testsuite name="ABC" tests="2" failures="1" skipped="0" time="410.63">
          <testcase classname="ABC/test_1" name="test_1" time="0.02"/>
          <testcase classname="ABC/test_2" name="test_2" time="0.02">
            <failure><![CDATA[ABC/test_2 output goes here]]></failure>
          </testcase>
          </testsuite>
          <testsuite name="DEF" tests="2" failures="1" skipped="1" time="410.63">
          <testcase classname="DEF/test_1" name="test_1" time="0.02">
            <skipped message="reason"/>
          </testcase>
          <testcase classname="DEF/test_2" name="test_2" time="0.02">
            <failure><![CDATA[DEF/test_2 output goes here]]></failure>
          </testcase>
          </testsuite>
          </testsuites>"""
                        )
                    )
                ],
            ),
            self.MULTI_SUITE_OUTPUT,
        )

    def test_report_multiple_files_multiple_testsuites(self):
        self.assertEqual(
            generate_test_report_lib.generate_report(
                "ABC and DEF",
                1,
                [
                    junit_from_xml(
                        dedent(
                            """\
          <?xml version="1.0" encoding="UTF-8"?>
          <testsuites time="8.89">
          <testsuite name="ABC" tests="2" failures="1" skipped="0" time="410.63">
          <testcase classname="ABC/test_1" name="test_1" time="0.02"/>
          <testcase classname="ABC/test_2" name="test_2" time="0.02">
            <failure><![CDATA[ABC/test_2 output goes here]]></failure>
          </testcase>
          </testsuite>
          </testsuites>"""
                        )
                    ),
                    junit_from_xml(
                        dedent(
                            """\
          <?xml version="1.0" encoding="UTF-8"?>
          <testsuites time="8.89">
          <testsuite name="DEF" tests="2" failures="1" skipped="1" time="410.63">
          <testcase classname="DEF/test_1" name="test_1" time="0.02">
            <skipped message="reason"/>
          </testcase>
          <testcase classname="DEF/test_2" name="test_2" time="0.02">
            <failure><![CDATA[DEF/test_2 output goes here]]></failure>
          </testcase>
          </testsuite>
          </testsuites>"""
                        )
                    ),
                ],
            ),
            self.MULTI_SUITE_OUTPUT,
        )

    def test_report_dont_list_failures(self):
        self.assertEqual(
            generate_test_report_lib.generate_report(
                "Foo",
                1,
                [
                    junit_from_xml(
                        dedent(
                            """\
          <?xml version="1.0" encoding="UTF-8"?>
          <testsuites time="0.02">
          <testsuite name="Bar" tests="1" failures="1" skipped="0" time="0.02">
          <testcase classname="Bar/test_1" name="test_1" time="0.02">
            <failure><![CDATA[Output goes here]]></failure>
          </testcase>
          </testsuite>
          </testsuites>"""
                        )
                    )
                ],
                list_failures=False,
            ),
            (
                dedent(
                    """\
          # Foo

          * 1 test failed

          Failed tests and their output was too large to report. Download the build's log file to see the details."""
                ),
                "error",
            ),
        )

    def test_report_dont_list_failures_link_to_log(self):
        self.assertEqual(
            generate_test_report_lib.generate_report(
                "Foo",
                1,
                [
                    junit_from_xml(
                        dedent(
                            """\
          <?xml version="1.0" encoding="UTF-8"?>
          <testsuites time="0.02">
          <testsuite name="Bar" tests="1" failures="1" skipped="0" time="0.02">
          <testcase classname="Bar/test_1" name="test_1" time="0.02">
            <failure><![CDATA[Output goes here]]></failure>
          </testcase>
          </testsuite>
          </testsuites>"""
                        )
                    )
                ],
                list_failures=False,
                buildkite_info={
                    "BUILDKITE_ORGANIZATION_SLUG": "organization_slug",
                    "BUILDKITE_PIPELINE_SLUG": "pipeline_slug",
                    "BUILDKITE_BUILD_NUMBER": "build_number",
                    "BUILDKITE_JOB_ID": "job_id",
                },
            ),
            (
                dedent(
                    """\
          # Foo

          * 1 test failed

          Failed tests and their output was too large to report. [Download](https://buildkite.com/organizations/organization_slug/pipelines/pipeline_slug/builds/build_number/jobs/job_id/download.txt) the build's log file to see the details."""
                ),
                "error",
            ),
        )

    def test_report_size_limit(self):
        self.assertEqual(
            generate_test_report_lib.generate_report(
                "Foo",
                1,
                [
                    junit_from_xml(
                        dedent(
                            """\
          <?xml version="1.0" encoding="UTF-8"?>
          <testsuites time="0.02">
          <testsuite name="Bar" tests="1" failures="1" skipped="0" time="0.02">
          <testcase classname="Bar/test_1" name="test_1" time="0.02">
            <failure><![CDATA[Some long output goes here...]]></failure>
          </testcase>
          </testsuite>
          </testsuites>"""
                        )
                    )
                ],
                size_limit=128,
            ),
            (
                dedent(
                    """\
          # Foo

          * 1 test failed

          Failed tests and their output was too large to report. Download the build's log file to see the details."""
                ),
                "error",
            ),
        )

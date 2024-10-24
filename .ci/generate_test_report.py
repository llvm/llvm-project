# Script to parse many JUnit XML result files and send a report to the buildkite
# agent as an annotation.
#
# To run the unittests:
# python3 -m unittest discover -p generate_test_report.py

import argparse
import unittest
from io import StringIO
from junitparser import JUnitXml, Failure
from textwrap import dedent
from subprocess import check_call


def junit_from_xml(xml):
    return JUnitXml.fromfile(StringIO(xml))


class TestReports(unittest.TestCase):
    def test_title_only(self):
        self.assertEqual(_generate_report("Foo", []), ("", None))

    def test_no_tests_in_testsuite(self):
        self.assertEqual(
            _generate_report(
                "Foo",
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
            _generate_report(
                "Foo",
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

    def test_report_single_file_single_testsuite(self):
        self.assertEqual(
            _generate_report(
                "Foo",
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

          ## Failed tests
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

        ## Failed tests
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
            _generate_report(
                "ABC and DEF",
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
            _generate_report(
                "ABC and DEF",
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


def _generate_report(title, junit_objects):
    style = None

    if not junit_objects:
        return ("", style)

    failures = {}
    tests_run = 0
    tests_skipped = 0
    tests_failed = 0

    for results in junit_objects:
        for testsuite in results:
            tests_run += testsuite.tests
            tests_skipped += testsuite.skipped
            tests_failed += testsuite.failures

            for test in testsuite:
                if (
                    not test.is_passed
                    and test.result
                    and isinstance(test.result[0], Failure)
                ):
                    if failures.get(testsuite.name) is None:
                        failures[testsuite.name] = []
                    failures[testsuite.name].append(
                        (test.classname + "/" + test.name, test.result[0].text)
                    )

    if not tests_run:
        return ("", style)

    style = "error" if tests_failed else "success"
    report = [f"# {title}", ""]

    tests_passed = tests_run - tests_skipped - tests_failed

    def plural(num_tests):
        return "test" if num_tests == 1 else "tests"

    if tests_passed:
        report.append(f"* {tests_passed} {plural(tests_passed)} passed")
    if tests_skipped:
        report.append(f"* {tests_skipped} {plural(tests_skipped)} skipped")
    if tests_failed:
        report.append(f"* {tests_failed} {plural(tests_failed)} failed")

    if failures:
        report.extend(["", "## Failed tests", "(click to see output)"])
        for testsuite_name, failures in failures.items():
            report.extend(["", f"### {testsuite_name}"])
            for name, output in failures:
                report.extend(
                    [
                        "<details>",
                        f"<summary>{name}</summary>",
                        "",
                        "```",
                        output,
                        "```",
                        "</details>",
                    ]
                )

    return "\n".join(report), style


def generate_report(title, junit_files):
    return _generate_report(title, [JUnitXml.fromfile(p) for p in junit_files])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "title", help="Title of the test report, without Markdown formatting."
    )
    parser.add_argument("context", help="Annotation context to write to.")
    parser.add_argument("junit_files", help="Paths to JUnit report files.", nargs="*")
    args = parser.parse_args()

    report, style = generate_report(args.title, args.junit_files)
    check_call(
        [
            "buildkite-agent",
            "annotate",
            "--context",
            args.context,
            "--style",
            style,
            report,
        ]
    )

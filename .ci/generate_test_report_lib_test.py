# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# To run these tests:
# python -m unittest generate_test_report_lib_test.py

import unittest
from io import StringIO
from textwrap import dedent
import tempfile
import os

from junitparser import JUnitXml

import generate_test_report_lib


def junit_from_xml(xml):
    return JUnitXml.fromfile(StringIO(xml))


class TestReports(unittest.TestCase):
    def test_find_failure_ninja_logs(self):
        failures = generate_test_report_lib.find_failure_in_ninja_logs(
            [
                [
                    "[1/5] test/1.stamp",
                    "[2/5] test/2.stamp",
                    "[3/5] test/3.stamp",
                    "[4/5] test/4.stamp",
                    "FAILED: touch test/4.stamp",
                    "Wow! This system is really broken!",
                    "[5/5] test/5.stamp",
                ],
            ]
        )
        self.assertEqual(len(failures), 1)
        self.assertEqual(
            failures[0],
            (
                "test/4.stamp",
                dedent(
                    """\
                    FAILED: touch test/4.stamp
                    Wow! This system is really broken!"""
                ),
            ),
        )

    def test_no_failure_ninja_log(self):
        failures = generate_test_report_lib.find_failure_in_ninja_logs(
            [
                [
                    "[1/3] test/1.stamp",
                    "[2/3] test/2.stamp",
                    "[3/3] test/3.stamp",
                ]
            ]
        )
        self.assertEqual(failures, [])

    def test_ninja_log_end(self):
        failures = generate_test_report_lib.find_failure_in_ninja_logs(
            [
                [
                    "[1/3] test/1.stamp",
                    "[2/3] test/2.stamp",
                    "[3/3] test/3.stamp",
                    "FAILED: touch test/3.stamp",
                    "Wow! This system is really broken!",
                    "ninja: build stopped: subcommand failed.",
                ]
            ]
        )
        self.assertEqual(len(failures), 1)
        self.assertEqual(
            failures[0],
            (
                "test/3.stamp",
                dedent(
                    """\
                    FAILED: touch test/3.stamp
                    Wow! This system is really broken!"""
                ),
            ),
        )

    def test_ninja_log_multiple_failures(self):
        failures = generate_test_report_lib.find_failure_in_ninja_logs(
            [
                [
                    "[1/5] test/1.stamp",
                    "[2/5] test/2.stamp",
                    "FAILED: touch test/2.stamp",
                    "Wow! This system is really broken!",
                    "[3/5] test/3.stamp",
                    "[4/5] test/4.stamp",
                    "FAILED: touch test/4.stamp",
                    "Wow! This system is maybe broken!",
                    "[5/5] test/5.stamp",
                ]
            ]
        )
        self.assertEqual(len(failures), 2)
        self.assertEqual(
            failures[0],
            (
                "test/2.stamp",
                dedent(
                    """\
                    FAILED: touch test/2.stamp
                    Wow! This system is really broken!"""
                ),
            ),
        )
        self.assertEqual(
            failures[1],
            (
                "test/4.stamp",
                dedent(
                    """\
                    FAILED: touch test/4.stamp
                    Wow! This system is maybe broken!"""
                ),
            ),
        )

    def test_title_only(self):
        self.assertEqual(
            generate_test_report_lib.generate_report("Foo", 0, [], []),
            dedent(
                """\
                # Foo

                The build succeeded and no tests ran. This is expected in some build configurations."""
            ),
        )

    def test_title_only_failure(self):
        self.assertEqual(
            generate_test_report_lib.generate_report("Foo", 1, [], []),
            dedent(
                """\
            # Foo

            The build failed before running any tests. Detailed information about the build failure could not be automatically obtained.

            Download the build's log file to see the details.

            If these failures are unrelated to your changes (for example tests are broken or flaky at HEAD), please open an issue at https://github.com/llvm/llvm-project/issues and add the `infrastructure` label."""
            ),
        )

    def test_title_only_failure_ninja_log(self):
        self.assertEqual(
            generate_test_report_lib.generate_report(
                "Foo",
                1,
                [],
                [
                    [
                        "[1/5] test/1.stamp",
                        "[2/5] test/2.stamp",
                        "[3/5] test/3.stamp",
                        "[4/5] test/4.stamp",
                        "FAILED: test/4.stamp",
                        "touch test/4.stamp",
                        "Wow! Risk!",
                        "[5/5] test/5.stamp",
                    ]
                ],
            ),
            dedent(
                """\
            # Foo

            The build failed before running any tests. Click on a failure below to see the details.

            <details>
            <summary>test/4.stamp</summary>

            ```
            FAILED: test/4.stamp
            touch test/4.stamp
            Wow! Risk!
            ```
            </details>
            
            If these failures are unrelated to your changes (for example tests are broken or flaky at HEAD), please open an issue at https://github.com/llvm/llvm-project/issues and add the `infrastructure` label."""
            ),
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
                [],
            ),
            dedent(
                """\
                # Foo

                The build failed before running any tests. Detailed information about the build failure could not be automatically obtained.

                Download the build's log file to see the details.

                If these failures are unrelated to your changes (for example tests are broken or flaky at HEAD), please open an issue at https://github.com/llvm/llvm-project/issues and add the `infrastructure` label."""
            ),
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
                [],
            ),
            (
                dedent(
                    """\
              # Foo

              * 1 test passed"""
                )
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
                [],
            ),
            (
                dedent(
                    """\
              # Foo

              * 1 test passed

              All tests passed but another part of the build **failed**. Information about the build failure could not be automatically obtained.

              Download the build's log file to see the details.
              
              If these failures are unrelated to your changes (for example tests are broken or flaky at HEAD), please open an issue at https://github.com/llvm/llvm-project/issues and add the `infrastructure` label."""
                )
            ),
        )

    def test_no_failures_build_failed_ninja_log(self):
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
                [
                    [
                        "[1/5] test/1.stamp",
                        "[2/5] test/2.stamp",
                        "[3/5] test/3.stamp",
                        "[4/5] test/4.stamp",
                        "FAILED: test/4.stamp",
                        "touch test/4.stamp",
                        "Wow! Close To You!",
                        "[5/5] test/5.stamp",
                    ]
                ],
            ),
            (
                dedent(
                    """\
                    # Foo

                    * 1 test passed

                    All tests passed but another part of the build **failed**. Click on a failure below to see the details.

                    <details>
                    <summary>test/4.stamp</summary>

                    ```
                    FAILED: test/4.stamp
                    touch test/4.stamp
                    Wow! Close To You!
                    ```
                    </details>

                    If these failures are unrelated to your changes (for example tests are broken or flaky at HEAD), please open an issue at https://github.com/llvm/llvm-project/issues and add the `infrastructure` label."""
                )
            ),
        )

    def test_no_failures_multiple_build_failed_ninja_log(self):
        test = generate_test_report_lib.generate_report(
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
            [
                [
                    "[1/5] test/1.stamp",
                    "[2/5] test/2.stamp",
                    "FAILED: touch test/2.stamp",
                    "Wow! Be Kind!",
                    "[3/5] test/3.stamp",
                    "[4/5] test/4.stamp",
                    "FAILED: touch test/4.stamp",
                    "Wow! I Dare You!",
                    "[5/5] test/5.stamp",
                ]
            ],
        )
        print(test)
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
                [
                    [
                        "[1/5] test/1.stamp",
                        "[2/5] test/2.stamp",
                        "FAILED: touch test/2.stamp",
                        "Wow! Be Kind!",
                        "[3/5] test/3.stamp",
                        "[4/5] test/4.stamp",
                        "FAILED: touch test/4.stamp",
                        "Wow! I Dare You!",
                        "[5/5] test/5.stamp",
                    ]
                ],
            ),
            (
                dedent(
                    """\
                    # Foo

                    * 1 test passed

                    All tests passed but another part of the build **failed**. Click on a failure below to see the details.

                    <details>
                    <summary>test/2.stamp</summary>

                    ```
                    FAILED: touch test/2.stamp
                    Wow! Be Kind!
                    ```
                    </details>
                    <details>
                    <summary>test/4.stamp</summary>

                    ```
                    FAILED: touch test/4.stamp
                    Wow! I Dare You!
                    ```
                    </details>

                    If these failures are unrelated to your changes (for example tests are broken or flaky at HEAD), please open an issue at https://github.com/llvm/llvm-project/issues and add the `infrastructure` label."""
                )
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
                [],
            ),
            (
                dedent(
                    """\
          # Foo

          * 1 test passed
          * 1 test skipped
          * 2 tests failed

          ## Failed Tests
          (click on a test name to see its output)

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
          </details>
          
          If these failures are unrelated to your changes (for example tests are broken or flaky at HEAD), please open an issue at https://github.com/llvm/llvm-project/issues and add the `infrastructure` label."""
                )
            ),
        )

    MULTI_SUITE_OUTPUT = dedent(
        """\
        # ABC and DEF

        * 1 test passed
        * 1 test skipped
        * 2 tests failed

        ## Failed Tests
        (click on a test name to see its output)

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
        </details>
        
        If these failures are unrelated to your changes (for example tests are broken or flaky at HEAD), please open an issue at https://github.com/llvm/llvm-project/issues and add the `infrastructure` label."""
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
                [],
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
                [],
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
                [],
                list_failures=False,
            ),
            (
                dedent(
                    """\
          # Foo

          * 1 test failed

          Failed tests and their output was too large to report. Download the build's log file to see the details.
          
          If these failures are unrelated to your changes (for example tests are broken or flaky at HEAD), please open an issue at https://github.com/llvm/llvm-project/issues and add the `infrastructure` label."""
                )
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
                [],
                list_failures=False,
            ),
            (
                dedent(
                    """\
          # Foo

          * 1 test failed

          Failed tests and their output was too large to report. Download the build's log file to see the details.
          
          If these failures are unrelated to your changes (for example tests are broken or flaky at HEAD), please open an issue at https://github.com/llvm/llvm-project/issues and add the `infrastructure` label."""
                )
            ),
        )

    def test_report_size_limit(self):
        test_output = "f" * 1000
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
            <failure><![CDATA[{output}]]></failure>
          </testcase>
          </testsuite>
          </testsuites>""".format(
                                output=test_output
                            )
                        )
                    )
                ],
                [],
                size_limit=512,
            ),
            (
                dedent(
                    """\
          # Foo

          * 1 test failed

          Failed tests and their output was too large to report. Download the build's log file to see the details.
          
          If these failures are unrelated to your changes (for example tests are broken or flaky at HEAD), please open an issue at https://github.com/llvm/llvm-project/issues and add the `infrastructure` label."""
                )
            ),
        )

    def test_generate_report_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            junit_xml_file = os.path.join(temp_dir, "junit.xml")
            with open(junit_xml_file, "w") as junit_xml_handle:
                junit_xml_handle.write(
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
            ninja_log_file = os.path.join(temp_dir, "ninja.log")
            with open(ninja_log_file, "w") as ninja_log_handle:
                ninja_log_handle.write(
                    dedent(
                        """\
                        [1/5] test/1.stamp
                        [2/5] test/2.stamp
                        [3/5] test/3.stamp
                        [4/5] test/4.stamp
                        FAILED: test/4.stamp
                        touch test/4.stamp
                        Wow! That's so True!
                        [5/5] test/5.stamp"""
                    )
                )
            self.assertEqual(
                generate_test_report_lib.generate_report_from_files(
                    "Foo", 1, [junit_xml_file, ninja_log_file]
                ),
                dedent(
                    """\
                    # Foo

                    * 1 test passed

                    All tests passed but another part of the build **failed**. Click on a failure below to see the details.

                    <details>
                    <summary>test/4.stamp</summary>

                    ```
                    FAILED: test/4.stamp
                    touch test/4.stamp
                    Wow! That's so True!
                    ```
                    </details>

                    If these failures are unrelated to your changes (for example tests are broken or flaky at HEAD), please open an issue at https://github.com/llvm/llvm-project/issues and add the `infrastructure` label."""
                ),
            )

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Library to parse JUnit XML files and return a markdown report."""

from junitparser import JUnitXml, Failure

SEE_BUILD_FILE_STR = "Download the build's log file to see the details."
UNRELATED_FAILURES_STR = (
    "If these failures are unrelated to your changes (for example "
    "tests are broken or flaky at HEAD), please open an issue at "
    "https://github.com/llvm/llvm-project/issues and add the "
    "`infrastructure` label."
)
# The maximum number of lines to pull from a ninja failure.
NINJA_LOG_SIZE_THRESHOLD = 500


def _parse_ninja_log(ninja_log: list[str]) -> list[tuple[str, str]]:
    """Parses an individual ninja log."""
    failures = []
    index = 0
    while index < len(ninja_log):
        while index < len(ninja_log) and not ninja_log[index].startswith("FAILED:"):
            index += 1
        if index == len(ninja_log):
            # We hit the end of the log without finding a build failure, go to
            # the next log.
            return failures
        # We are trying to parse cases like the following:
        #
        # [4/5] test/4.stamp
        # FAILED: touch test/4.stamp
        # touch test/4.stamp
        #
        # index will point to the line that starts with Failed:. The progress
        # indicator is the line before this ([4/5] test/4.stamp) and contains a pretty
        # printed version of the target being built (test/4.stamp). We use this line
        # and remove the progress information to get a succinct name for the target.
        failing_action = ninja_log[index - 1].split("] ")[1]
        failure_log = []
        while (
            index < len(ninja_log)
            and not ninja_log[index].startswith("[")
            and not ninja_log[index].startswith("ninja: build stopped:")
            and len(failure_log) < NINJA_LOG_SIZE_THRESHOLD
        ):
            failure_log.append(ninja_log[index])
            index += 1
        failures.append((failing_action, "\n".join(failure_log)))
    return failures


def find_failure_in_ninja_logs(ninja_logs: list[list[str]]) -> list[tuple[str, str]]:
    """Extracts failure messages from ninja output.

    This function takes stdout/stderr from ninja in the form of a list of files
    represented as a list of lines. This function then returns tuples containing
    the name of the target and the error message.

    Args:
      ninja_logs: A list of files in the form of a list of lines representing the log
        files captured from ninja.

    Returns:
      A list of tuples. The first string is the name of the target that failed. The
      second string is the error message.
    """
    failures = []
    for ninja_log in ninja_logs:
        log_failures = _parse_ninja_log(ninja_log)
        failures.extend(log_failures)
    return failures


def _format_ninja_failures(ninja_failures: list[tuple[str, str]]) -> list[str]:
    """Formats ninja failures into summary views for the report."""
    output = []
    for build_failure in ninja_failures:
        failed_action, failure_message = build_failure
        output.extend(
            [
                "<details>",
                f"<summary>{failed_action}</summary>",
                "",
                "```",
                failure_message,
                "```",
                "</details>",
            ]
        )
    return output


# Set size_limit to limit the byte size of the report. The default is 1MB as this
# is the most that can be put into an annotation. If the generated report exceeds
# this limit and failures are listed, it will be generated again without failures
# listed. This minimal report will always fit into an annotation.
# If include failures is False, total number of test will be reported but their names
# and output will not be.
def generate_report(
    title,
    return_code,
    junit_objects,
    ninja_logs: list[list[str]],
    size_limit=1024 * 1024,
    list_failures=True,
):
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

    report = [f"# {title}", ""]

    if tests_run == 0:
        if return_code == 0:
            report.extend(
                [
                    "The build succeeded and no tests ran. This is expected in some "
                    "build configurations."
                ]
            )
        else:
            ninja_failures = find_failure_in_ninja_logs(ninja_logs)
            if not ninja_failures:
                report.extend(
                    [
                        "The build failed before running any tests. Detailed "
                        "information about the build failure could not be "
                        "automatically obtained.",
                        "",
                        SEE_BUILD_FILE_STR,
                        "",
                        UNRELATED_FAILURES_STR,
                    ]
                )
            else:
                report.extend(
                    [
                        "The build failed before running any tests. Click on a "
                        "failure below to see the details.",
                        "",
                    ]
                )
                report.extend(_format_ninja_failures(ninja_failures))
                report.extend(
                    [
                        "",
                        UNRELATED_FAILURES_STR,
                    ]
                )
        return "\n".join(report)

    tests_passed = tests_run - tests_skipped - tests_failed

    def plural(num_tests):
        return "test" if num_tests == 1 else "tests"

    if tests_passed:
        report.append(f"* {tests_passed} {plural(tests_passed)} passed")
    if tests_skipped:
        report.append(f"* {tests_skipped} {plural(tests_skipped)} skipped")
    if tests_failed:
        report.append(f"* {tests_failed} {plural(tests_failed)} failed")

    if not list_failures:
        report.extend(
            [
                "",
                "Failed tests and their output was too large to report. "
                + SEE_BUILD_FILE_STR,
            ]
        )
    elif failures:
        report.extend(
            ["", "## Failed Tests", "(click on a test name to see its output)"]
        )

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
    elif return_code != 0:
        # No tests failed but the build was in a failed state. Bring this to the user's
        # attention.
        ninja_failures = find_failure_in_ninja_logs(ninja_logs)
        if not ninja_failures:
            report.extend(
                [
                    "",
                    "All tests passed but another part of the build **failed**. "
                    "Information about the build failure could not be automatically "
                    "obtained.",
                    "",
                    SEE_BUILD_FILE_STR,
                ]
            )
        else:
            report.extend(
                [
                    "",
                    "All tests passed but another part of the build **failed**. Click on "
                    "a failure below to see the details.",
                    "",
                ]
            )
            report.extend(_format_ninja_failures(ninja_failures))

    if failures or return_code != 0:
        report.extend(["", UNRELATED_FAILURES_STR])

    report = "\n".join(report)
    if len(report.encode("utf-8")) > size_limit:
        return generate_report(
            title,
            return_code,
            junit_objects,
            size_limit,
            list_failures=False,
        )

    return report


def generate_report_from_files(title, return_code, build_log_files):
    junit_files = [
        junit_file for junit_file in build_log_files if junit_file.endswith(".xml")
    ]
    ninja_log_files = [
        ninja_log for ninja_log in build_log_files if ninja_log.endswith(".log")
    ]
    ninja_logs = []
    for ninja_log_file in ninja_log_files:
        with open(ninja_log_file, "r") as ninja_log_file_handle:
            ninja_logs.append(
                [log_line.strip() for log_line in ninja_log_file_handle.readlines()]
            )
    return generate_report(
        title, return_code, [JUnitXml.fromfile(p) for p in junit_files], ninja_logs
    )

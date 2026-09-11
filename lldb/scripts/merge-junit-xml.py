#!/usr/bin/env python3

"""
Usage: merge-junit-xml.py [options] <first.xml> <second.xml>

Merge two JUnit XML reports into a single report on stdout.

Tests are identified by their test suite, class name and test name. When the
same test is present in both reports, the more interesting result wins, in the
order

    skipped < passed < failed

so that a test failing in either report is reported as failing. Results that
are equally interesting are taken from the first report, which keeps the more
detailed skip reasons of a full test run when merging in a partial one.

With --ignore-lhs-failures-matching, failures in the first (left hand side)
report whose message or output matches the given regular expression lose
against any result from the second report, but still win against a test that
the second report skipped or does not mention at all. This is meant for merging
the results of a test run that is known to produce spurious failures with the
results of re-running the failed tests: a spurious failure is replaced by the
result of the re-run, while one that never got re-run stays a failure.

The failure output of the merged tests is copied verbatim, but character data
that was wrapped in a CDATA section is escaped instead.

The exit status is 1 if the merged report contains any failure, and 2 if a
report could not be read, so that re-running the failed tests of a test run can
be scripted as

    lit ... --xunit-xml-output=results.xml || \
        lit ... --filter-failed --xunit-xml-output=rerun.xml
    merge-junit-xml.py results.xml rerun.xml -o results.xml \
        --ignore-lhs-failures-matching "..."
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET

# How interesting a test result is. See the module docstring for how this is
# used to pick a winner.
SKIPPED, IGNORED_FAILURE, PASSED, FAILED = range(4)


def parse_report(path):
    """Return the test suites of a JUnit report and its total time."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        print("error: %s: %s" % (path, error), file=sys.stderr)
        sys.exit(2)
    time = float(root.get("time") or 0.0)
    if root.tag == "testsuite":
        return [root], time
    return list(root.iter("testsuite")), time


def result_of(testcase):
    if testcase.find("failure") is not None or testcase.find("error") is not None:
        return FAILED
    if testcase.find("skipped") is not None:
        return SKIPPED
    return PASSED


def failure_output(testcase):
    """Return the messages and output of all failures of a test."""
    output = []
    for element in list(testcase.iter("failure")) + list(testcase.iter("error")):
        output.append(element.get("message") or "")
        output.append(element.text or "")
    return "\n".join(output)


class TestSuite:
    """The merged test cases of one test suite, in the order they were added."""

    def __init__(self, element):
        # Keep the attributes of the suite the test cases came from first; the
        # counts among them are recomputed when the suite is written out.
        self.element = element
        self.testcases = {}
        self.ranks = {}

    def add(self, testcase, rank):
        key = (testcase.get("classname"), testcase.get("name"))
        if key in self.testcases and rank <= self.ranks[key]:
            return
        self.testcases[key] = testcase
        self.ranks[key] = rank

    def to_element(self):
        element = ET.Element("testsuite", dict(self.element.attrib))
        # Anything that is not a test case, e.g. <properties>, is passed through.
        for child in self.element:
            if child.tag != "testcase":
                element.append(child)
        failures = skipped = 0
        time = 0.0
        for testcase in self.testcases.values():
            element.append(testcase)
            result = result_of(testcase)
            failures += result == FAILED
            skipped += result == SKIPPED
            time += float(testcase.get("time") or 0.0)
        element.set("tests", str(len(self.testcases)))
        element.set("failures", str(failures))
        element.set("skipped", str(skipped))
        element.set("time", "%.2f" % time)
        return element


def merge(suites, report, ignore=None, verbose=False):
    """Merge the test suites of one report into the merged suites so far."""
    for suite in report:
        name = suite.get("name")
        if name not in suites:
            suites[name] = TestSuite(suite)
        merged = suites[name]
        for testcase in suite.findall("testcase"):
            rank = result_of(testcase)
            if ignore and rank == FAILED and ignore.search(failure_output(testcase)):
                rank = IGNORED_FAILURE
                if verbose:
                    print(
                        "ignoring failure of %s :: %s/%s"
                        % (name, testcase.get("classname"), testcase.get("name")),
                        file=sys.stderr,
                    )
            merged.add(testcase, rank)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("first", help="the JUnit XML report to merge into")
    parser.add_argument("second", help="the JUnit XML report to merge from")
    parser.add_argument(
        "--ignore-lhs-failures-matching",
        metavar="REGEX",
        type=re.compile,
        help="let failures in the first (left hand side) report whose message "
        "or output matches REGEX lose against a result from the second report",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="write the merged report to FILE instead of stdout",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="report what is being merged on stderr",
    )
    args = parser.parse_args()

    first, first_time = parse_report(args.first)
    second, second_time = parse_report(args.second)

    suites = {}
    merge(suites, first, args.ignore_lhs_failures_matching, args.verbose)
    merge(suites, second, verbose=args.verbose)

    root = ET.Element("testsuites", {"time": "%.2f" % (first_time + second_time)})
    for suite in suites.values():
        root.append(suite.to_element())

    if hasattr(ET, "indent"):
        ET.indent(root, space="")
    tree = ET.ElementTree(root)
    if args.output:
        tree.write(args.output, encoding="UTF-8", xml_declaration=True)
    else:
        tree.write(sys.stdout.buffer, encoding="UTF-8", xml_declaration=True)

    failures = sum(int(suite.get("failures")) for suite in root)
    if args.verbose:
        tests = sum(int(suite.get("tests")) for suite in root)
        print(
            "merged report: %d tests, %d failures" % (tests, failures), file=sys.stderr
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

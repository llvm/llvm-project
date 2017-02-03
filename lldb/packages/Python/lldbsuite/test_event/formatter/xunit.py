"""
                     The LLVM Compiler Infrastructure

This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.

Provides an xUnit ResultsFormatter for integrating the LLDB
test suite with the Jenkins xUnit aggregator and other xUnit-compliant
test output processors.
"""
from __future__ import absolute_import
from __future__ import print_function

# System modules
import re
import sys
import xml.sax.saxutils

# Third-party modules
import six

# Local modules
from ..event_builder import EventBuilder
from ..build_exception import BuildError
from .results_formatter import ResultsFormatter


class XunitFormatter(ResultsFormatter):
    """Provides xUnit-style formatted output.
    """

    # Result mapping arguments
    RM_IGNORE = 'ignore'
    RM_SUCCESS = 'success'
    RM_FAILURE = 'failure'
    RM_PASSTHRU = 'passthru'

    @staticmethod
    def _build_illegal_xml_regex():
        """Constructs a regex to match all illegal xml characters.

        Expects to be used against a unicode string."""
        # Construct the range pairs of invalid unicode characters.
        illegal_chars_u = [
            (0x00, 0x08), (0x0B, 0x0C), (0x0E, 0x1F), (0x7F, 0x84),
            (0x86, 0x9F), (0xFDD0, 0xFDDF), (0xFFFE, 0xFFFF)]

        # For wide builds, we have more.
        if sys.maxunicode >= 0x10000:
            illegal_chars_u.extend(
                [(0x1FFFE, 0x1FFFF), (0x2FFFE, 0x2FFFF), (0x3FFFE, 0x3FFFF),
                 (0x4FFFE, 0x4FFFF), (0x5FFFE, 0x5FFFF), (0x6FFFE, 0x6FFFF),
                 (0x7FFFE, 0x7FFFF), (0x8FFFE, 0x8FFFF), (0x9FFFE, 0x9FFFF),
                 (0xAFFFE, 0xAFFFF), (0xBFFFE, 0xBFFFF), (0xCFFFE, 0xCFFFF),
                 (0xDFFFE, 0xDFFFF), (0xEFFFE, 0xEFFFF), (0xFFFFE, 0xFFFFF),
                 (0x10FFFE, 0x10FFFF)])

        # Build up an array of range expressions.
        illegal_ranges = [
            "%s-%s" % (six.unichr(low), six.unichr(high))
            for (low, high) in illegal_chars_u]

        # Compile the regex
        return re.compile(six.u('[%s]') % six.u('').join(illegal_ranges))

    @staticmethod
    def _quote_attribute(text):
        """Returns the given text in a manner safe for usage in an XML attribute.

        @param text the text that should appear within an XML attribute.
        @return the attribute-escaped version of the input text.
        """
        return xml.sax.saxutils.quoteattr(text)

    def _replace_invalid_xml(self, str_or_unicode):
        """Replaces invalid XML characters with a '?'.

        @param str_or_unicode a string to replace invalid XML
        characters within.  Can be unicode or not.  If not unicode,
        assumes it is a byte string in utf-8 encoding.

        @returns a utf-8-encoded byte string with invalid
        XML replaced with '?'.
        """
        # Get the content into unicode
        if isinstance(str_or_unicode, str):
            unicode_content = str_or_unicode.decode('utf-8')
        else:
            unicode_content = str_or_unicode
        return self.invalid_xml_re.sub(
            six.u('?'), unicode_content).encode('utf-8')

    @classmethod
    def arg_parser(cls):
        """@return arg parser used to parse formatter-specific options."""
        parser = super(XunitFormatter, cls).arg_parser()

        # These are valid choices for results mapping.
        results_mapping_choices = [
            XunitFormatter.RM_IGNORE,
            XunitFormatter.RM_SUCCESS,
            XunitFormatter.RM_FAILURE,
            XunitFormatter.RM_PASSTHRU]
        parser.add_argument(
            "--assert-on-unknown-events",
            action="store_true",
            help=('cause unknown test events to generate '
                  'a python assert.  Default is to ignore.'))
        parser.add_argument(
            "--ignore-skip-name",
            "-n",
            metavar='PATTERN',
            action="append",
            dest='ignore_skip_name_patterns',
            help=('a python regex pattern, where '
                  'any skipped test with a test method name where regex '
                  'matches (via search) will be ignored for xUnit test '
                  'result purposes.  Can be specified multiple times.'))
        parser.add_argument(
            "--ignore-skip-reason",
            "-r",
            metavar='PATTERN',
            action="append",
            dest='ignore_skip_reason_patterns',
            help=('a python regex pattern, where '
                  'any skipped test with a skip reason where the regex '
                  'matches (via search) will be ignored for xUnit test '
                  'result purposes.  Can be specified multiple times.'))
        parser.add_argument(
            "--xpass", action="store", choices=results_mapping_choices,
            default=XunitFormatter.RM_FAILURE,
            help=('specify mapping from unexpected success to jUnit/xUnit '
                  'result type'))
        parser.add_argument(
            "--xfail", action="store", choices=results_mapping_choices,
            default=XunitFormatter.RM_IGNORE,
            help=('specify mapping from expected failure to jUnit/xUnit '
                  'result type'))
        return parser

    @staticmethod
    def _build_regex_list_from_patterns(patterns):
        """Builds a list of compiled regular expressions from option value.

        @param patterns contains a list of regular expression
        patterns.

        @return list of compiled regular expressions, empty if no
        patterns provided.
        """
        regex_list = []
        if patterns is not None:
            for pattern in patterns:
                regex_list.append(re.compile(pattern))
        return regex_list

    def __init__(self, out_file, options, file_is_stream):
        """Initializes the XunitFormatter instance.
        @param out_file file-like object where formatted output is written.
        @param options specifies a dictionary of options for the
        formatter.
        """
        # Initialize the parent
        super(XunitFormatter, self).__init__(out_file, options, file_is_stream)
        self.text_encoding = "UTF-8"
        self.invalid_xml_re = XunitFormatter._build_illegal_xml_regex()
        self.total_test_count = 0
        self.ignore_skip_name_regexes = (
            XunitFormatter._build_regex_list_from_patterns(
                options.ignore_skip_name_patterns))
        self.ignore_skip_reason_regexes = (
            XunitFormatter._build_regex_list_from_patterns(
                options.ignore_skip_reason_patterns))

        self.elements = {
            "successes": [],
            "errors": [],
            "failures": [],
            "skips": [],
            "unexpected_successes": [],
            "expected_failures": [],
            "all": []
        }

        self.status_handlers = {
            EventBuilder.STATUS_SUCCESS: self._handle_success,
            EventBuilder.STATUS_FAILURE: self._handle_failure,
            EventBuilder.STATUS_ERROR: self._handle_error,
            EventBuilder.STATUS_SKIP: self._handle_skip,
            EventBuilder.STATUS_EXPECTED_FAILURE:
                self._handle_expected_failure,
            EventBuilder.STATUS_EXPECTED_TIMEOUT:
                self._handle_expected_timeout,
            EventBuilder.STATUS_UNEXPECTED_SUCCESS:
                self._handle_unexpected_success,
            EventBuilder.STATUS_EXCEPTIONAL_EXIT:
                self._handle_exceptional_exit,
            EventBuilder.STATUS_TIMEOUT:
                self._handle_timeout
        }

    RESULT_TYPES = {
        EventBuilder.TYPE_TEST_RESULT,
        EventBuilder.TYPE_JOB_RESULT}

    def handle_event(self, test_event):
        super(XunitFormatter, self).handle_event(test_event)

        event_type = test_event["event"]
        if event_type is None:
            return

        if event_type == "terminate":
            # Process all the final result events into their
            # XML counterparts.
            for result_event in self.result_events.values():
                self._process_test_result(result_event)
            self._finish_output()
        else:
            # This is an unknown event.
            if self.options.assert_on_unknown_events:
                raise Exception("unknown event type {} from {}\n".format(
                    event_type, test_event))

    def _handle_success(self, test_event):
        """Handles a test success.
        @param test_event the test event to handle.
        """
        result = self._common_add_testcase_entry(test_event)
        with self.lock:
            self.elements["successes"].append(result)

    def _handle_failure(self, test_event):
        """Handles a test failure.
        @param test_event the test event to handle.
        """
        message = self._replace_invalid_xml(test_event["issue_message"])
        backtrace = self._replace_invalid_xml(
            "".join(test_event.get("issue_backtrace", [])))

        result = self._common_add_testcase_entry(
            test_event,
            inner_content=(
                '<failure type={} message={}><![CDATA[{}]]></failure>'.format(
                    XunitFormatter._quote_attribute(test_event["issue_class"]),
                    XunitFormatter._quote_attribute(message),
                    backtrace)
            ))
        with self.lock:
            self.elements["failures"].append(result)

    def _handle_error_build(self, test_event):
        """Handles a test error.
        @param test_event the test event to handle.
        """
        message = self._replace_invalid_xml(test_event["issue_message"])
        build_issue_description = self._replace_invalid_xml(
            BuildError.format_build_error(
                test_event.get("build_command", "<None>"),
                test_event.get("build_error", "<None>")))

        result = self._common_add_testcase_entry(
            test_event,
            inner_content=(
                '<error type={} message={}><![CDATA[{}]]></error>'.format(
                    XunitFormatter._quote_attribute(test_event["issue_class"]),
                    XunitFormatter._quote_attribute(message),
                    build_issue_description)
            ))
        with self.lock:
            self.elements["errors"].append(result)

    def _handle_error_standard(self, test_event):
        """Handles a test error.
        @param test_event the test event to handle.
        """
        message = self._replace_invalid_xml(test_event["issue_message"])
        backtrace = self._replace_invalid_xml(
            "".join(test_event.get("issue_backtrace", [])))

        result = self._common_add_testcase_entry(
            test_event,
            inner_content=(
                '<error type={} message={}><![CDATA[{}]]></error>'.format(
                    XunitFormatter._quote_attribute(test_event["issue_class"]),
                    XunitFormatter._quote_attribute(message),
                    backtrace)
            ))
        with self.lock:
            self.elements["errors"].append(result)

    def _handle_error(self, test_event):
        if test_event.get("issue_phase", None) == "build":
            self._handle_error_build(test_event)
        else:
            self._handle_error_standard(test_event)

    def _handle_exceptional_exit(self, test_event):
        """Handles an exceptional exit.
        @param test_event the test method or job result event to handle.
        """
        if "test_name" in test_event:
            name = test_event["test_name"]
        else:
            name = test_event.get("test_filename", "<unknown test/filename>")

        message_text = "ERROR: {} ({}): {}".format(
            test_event.get("exception_code", 0),
            test_event.get("exception_description", ""),
            name)
        message = self._replace_invalid_xml(message_text)

        result = self._common_add_testcase_entry(
            test_event,
            inner_content=(
                '<error type={} message={}></error>'.format(
                    "exceptional_exit",
                    XunitFormatter._quote_attribute(message))
            ))
        with self.lock:
            self.elements["errors"].append(result)

    def _handle_timeout(self, test_event):
        """Handles a test method or job timeout.
        @param test_event the test method or job result event to handle.
        """
        if "test_name" in test_event:
            name = test_event["test_name"]
        else:
            name = test_event.get("test_filename", "<unknown test/filename>")

        message_text = "TIMEOUT: {}".format(name)
        message = self._replace_invalid_xml(message_text)

        result = self._common_add_testcase_entry(
            test_event,
            inner_content=(
                '<error type={} message={}></error>'.format(
                    "timeout",
                    XunitFormatter._quote_attribute(message))
            ))
        with self.lock:
            self.elements["errors"].append(result)

    @staticmethod
    def _ignore_based_on_regex_list(test_event, test_key, regex_list):
        """Returns whether to ignore a test event based on patterns.

        @param test_event the test event dictionary to check.
        @param test_key the key within the dictionary to check.
        @param regex_list a list of zero or more regexes.  May contain
        zero or more compiled regexes.

        @return True if any o the regex list match based on the
        re.search() method; false otherwise.
        """
        for regex in regex_list:
            match = regex.search(test_event.get(test_key, ''))
            if match:
                return True
        return False

    def _handle_skip(self, test_event):
        """Handles a skipped test.
        @param test_event the test event to handle.
        """

        # Are we ignoring this test based on test name?
        if XunitFormatter._ignore_based_on_regex_list(
                test_event, 'test_name', self.ignore_skip_name_regexes):
            return

        # Are we ignoring this test based on skip reason?
        if XunitFormatter._ignore_based_on_regex_list(
                test_event, 'skip_reason', self.ignore_skip_reason_regexes):
            return

        # We're not ignoring this test.  Process the skip.
        reason = self._replace_invalid_xml(test_event.get("skip_reason", ""))
        result = self._common_add_testcase_entry(
            test_event,
            inner_content='<skipped message={} />'.format(
                XunitFormatter._quote_attribute(reason)))
        with self.lock:
            self.elements["skips"].append(result)

    def _handle_expected_failure(self, test_event):
        """Handles a test that failed as expected.
        @param test_event the test event to handle.
        """
        if self.options.xfail == XunitFormatter.RM_PASSTHRU:
            # This is not a natively-supported junit/xunit
            # testcase mode, so it might fail a validating
            # test results viewer.
            if "bugnumber" in test_event:
                bug_id_attribute = 'bug-id={} '.format(
                    XunitFormatter._quote_attribute(test_event["bugnumber"]))
            else:
                bug_id_attribute = ''

            result = self._common_add_testcase_entry(
                test_event,
                inner_content=(
                    '<expected-failure {}type={} message={} />'.format(
                        bug_id_attribute,
                        XunitFormatter._quote_attribute(
                            test_event["issue_class"]),
                        XunitFormatter._quote_attribute(
                            test_event["issue_message"]))
                ))
            with self.lock:
                self.elements["expected_failures"].append(result)
        elif self.options.xfail == XunitFormatter.RM_SUCCESS:
            result = self._common_add_testcase_entry(
                test_event, test_name_suffix="XFAIL:")
            with self.lock:
                self.elements["successes"].append(result)
        elif self.options.xfail == XunitFormatter.RM_FAILURE:
            result = self._common_add_testcase_entry(
                test_event,
                inner_content='<failure type={} message={} />'.format(
                    XunitFormatter._quote_attribute(test_event["issue_class"]),
                    XunitFormatter._quote_attribute(
                        test_event["issue_message"])))
            with self.lock:
                self.elements["failures"].append(result)
        elif self.options.xfail == XunitFormatter.RM_IGNORE:
            pass
        else:
            raise Exception(
                "unknown xfail option: {}".format(self.options.xfail))

    @staticmethod
    def _handle_expected_timeout(test_event):
        """Handles expected_timeout.
        @param test_event the test event to handle.
        """
        # We don't do anything with expected timeouts, not even report.
        pass

    def _handle_unexpected_success(self, test_event):
        """Handles a test that passed but was expected to fail.
        @param test_event the test event to handle.
        """
        if self.options.xpass == XunitFormatter.RM_PASSTHRU:
            # This is not a natively-supported junit/xunit
            # testcase mode, so it might fail a validating
            # test results viewer.
            result = self._common_add_testcase_entry(
                test_event,
                inner_content="<unexpected-success />")
            with self.lock:
                self.elements["unexpected_successes"].append(result)
        elif self.options.xpass == XunitFormatter.RM_SUCCESS:
            # Treat the xpass as a success.
            result = self._common_add_testcase_entry(
                test_event, test_name_suffix="XPASS:")
            with self.lock:
                self.elements["successes"].append(result)
        elif self.options.xpass == XunitFormatter.RM_FAILURE:
            # Treat the xpass as a failure.
            if "bugnumber" in test_event:
                message = "unexpected success (bug_id:{})".format(
                    test_event["bugnumber"])
            else:
                message = "unexpected success (bug_id:none)"
            result = self._common_add_testcase_entry(
                test_event,
                inner_content='<failure type={} message={} />'.format(
                    XunitFormatter._quote_attribute("unexpected_success"),
                    XunitFormatter._quote_attribute(message)))
            with self.lock:
                self.elements["failures"].append(result)
        elif self.options.xpass == XunitFormatter.RM_IGNORE:
            # Ignore the xpass result as far as xUnit reporting goes.
            pass
        else:
            raise Exception("unknown xpass option: {}".format(
                self.options.xpass))

    def _process_test_result(self, test_event):
        """Processes the test_event known to be a test result.

        This categorizes the event appropriately and stores the data needed
        to generate the final xUnit report.  This method skips events that
        cannot be represented in xUnit output.
        """
        if "status" not in test_event:
            raise Exception("test event dictionary missing 'status' key")

        status = test_event["status"]
        if status not in self.status_handlers:
            raise Exception("test event status '{}' unsupported".format(
                status))

        # Call the status handler for the test result.
        self.status_handlers[status](test_event)

    def _common_add_testcase_entry(
            self,
            test_event,
            inner_content=None,
            test_name_suffix=None):
        """Registers a testcase result, and returns the text created.

        The caller is expected to manage failure/skip/success counts
        in some kind of appropriate way.  This call simply constructs
        the XML and appends the returned result to the self.all_results
        list.

        @param test_event the test event dictionary.

        @param inner_content if specified, gets included in the <testcase>
        inner section, at the point before stdout and stderr would be
        included.  This is where a <failure/>, <skipped/>, <error/>, etc.
        could go.

        @return the text of the xml testcase element.
        """

        # Get elapsed time.
        test_class = test_event.get("test_class", "<no_class>")
        test_name = test_event.get("test_name", "<no_test_method>")
        if test_name_suffix is not None:
            test_name = test_name_suffix + test_name
        event_time = test_event["event_time"]
        time_taken = self.elapsed_time_for_test(
            test_class, test_name, event_time)

        # Plumb in stdout/stderr once we shift over to only test results.
        test_stdout = ''
        test_stderr = ''

        # Formulate the output xml.
        if not inner_content:
            inner_content = ""
        result = (
            '<testcase classname="{}" name="{}" time="{:.3f}">'
            '{}{}{}</testcase>'.format(
                test_class,
                test_name,
                time_taken,
                inner_content,
                test_stdout,
                test_stderr))

        # Save the result, update total test count.
        with self.lock:
            self.total_test_count += 1
            self.elements["all"].append(result)

        return result

    def _finish_output_no_lock(self):
        """Flushes out the report of test executions to form valid xml output.

        xUnit output is in XML.  The reporting system cannot complete the
        formatting of the output without knowing when there is no more input.
        This call addresses notification of the completed test run and thus is
        when we can finish off the report output.
        """

        # Figure out the counts line for the testsuite.  If we have
        # been counting either unexpected successes or expected
        # failures, we'll output those in the counts, at the risk of
        # being invalidated by a validating test results viewer.
        # These aren't counted by default so they won't show up unless
        # the user specified a formatter option to include them.
        xfail_count = len(self.elements["expected_failures"])
        xpass_count = len(self.elements["unexpected_successes"])
        if xfail_count > 0 or xpass_count > 0:
            extra_testsuite_attributes = (
                ' expected-failures="{}"'
                ' unexpected-successes="{}"'.format(xfail_count, xpass_count))
        else:
            extra_testsuite_attributes = ""

        # Output the header.
        self.out_file.write(
            '<?xml version="1.0" encoding="{}"?>\n'
            '<testsuites>'
            '<testsuite name="{}" tests="{}" errors="{}" failures="{}" '
            'skip="{}"{}>\n'.format(
                self.text_encoding,
                "LLDB test suite",
                self.total_test_count,
                len(self.elements["errors"]),
                len(self.elements["failures"]),
                len(self.elements["skips"]),
                extra_testsuite_attributes))

        # Output each of the test result entries.
        for result in self.elements["all"]:
            self.out_file.write(result + '\n')

        # Close off the test suite.
        self.out_file.write('</testsuite></testsuites>\n')

    def _finish_output(self):
        """Finish writing output as all incoming events have arrived."""
        with self.lock:
            self._finish_output_no_lock()

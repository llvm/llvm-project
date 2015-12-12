"""
                     The LLVM Compiler Infrastructure

This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.

Provides classes used by the test results reporting infrastructure
within the LLDB test suite.
"""

from __future__ import print_function
from __future__ import absolute_import

# System modules
import argparse
import importlib
import inspect
import os
import pprint
import socket
import sys
import threading
import time
import traceback

# Third-party modules
import six
from six.moves import cPickle

# LLDB modules
from . import configuration


# Ignore method count on DTOs.
# pylint: disable=too-few-public-methods
class FormatterConfig(object):
    """Provides formatter configuration info to create_results_formatter()."""
    def __init__(self):
        self.filename = None
        self.port = None
        self.formatter_name = None
        self.formatter_options = None


# Ignore method count on DTOs.
# pylint: disable=too-few-public-methods
class CreatedFormatter(object):
    """Provides transfer object for returns from create_results_formatter()."""
    def __init__(self, formatter, cleanup_func):
        self.formatter = formatter
        self.cleanup_func = cleanup_func


def create_results_formatter(config):
    """Sets up a test results formatter.

    @param config an instance of FormatterConfig
    that indicates how to setup the ResultsFormatter.

    @return an instance of CreatedFormatter.
    """
    def create_socket(port):
        """Creates a socket to the localhost on the given port.

        @param port the port number of the listenering port on
        the localhost.

        @return (socket object, socket closing function)
        """
        def socket_closer(open_sock):
            """Close down an opened socket properly."""
            open_sock.shutdown(socket.SHUT_RDWR)
            open_sock.close()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("localhost", port))
        return (sock, lambda: socket_closer(sock))

    default_formatter_name = None
    results_file_object = None
    cleanup_func = None

    if config.filename:
        # Open the results file for writing.
        if config.filename == 'stdout':
            results_file_object = sys.stdout
            cleanup_func = None
        elif config.filename == 'stderr':
            results_file_object = sys.stderr
            cleanup_func = None
        else:
            results_file_object = open(config.filename, "w")
            cleanup_func = results_file_object.close
        default_formatter_name = (
            "lldbsuite.test.xunit_formatter.XunitFormatter")
    elif config.port:
        # Connect to the specified localhost port.
        results_file_object, cleanup_func = create_socket(config.port)
        default_formatter_name = (
            "lldbsuite.test.result_formatter.RawPickledFormatter")

    # If we have a results formatter name specified and we didn't specify
    # a results file, we should use stdout.
    if config.formatter_name is not None and results_file_object is None:
        # Use stdout.
        results_file_object = sys.stdout
        cleanup_func = None

    if results_file_object:
        # We care about the formatter.  Choose user-specified or, if
        # none specified, use the default for the output type.
        if config.formatter_name:
            formatter_name = config.formatter_name
        else:
            formatter_name = default_formatter_name

        # Create an instance of the class.
        # First figure out the package/module.
        components = formatter_name.split(".")
        module = importlib.import_module(".".join(components[:-1]))

        # Create the class name we need to load.
        cls = getattr(module, components[-1])

        # Handle formatter options for the results formatter class.
        formatter_arg_parser = cls.arg_parser()
        if config.formatter_options and len(config.formatter_options) > 0:
            command_line_options = config.formatter_options
        else:
            command_line_options = []

        formatter_options = formatter_arg_parser.parse_args(
            command_line_options)

        # Create the TestResultsFormatter given the processed options.
        results_formatter_object = cls(results_file_object, formatter_options)

        def shutdown_formatter():
            """Shuts down the formatter when it is no longer needed."""
            # Tell the formatter to write out anything it may have
            # been saving until the very end (e.g. xUnit results
            # can't complete its output until this point).
            results_formatter_object.send_terminate_as_needed()

            # And now close out the output file-like object.
            if cleanup_func is not None:
                cleanup_func()

        return CreatedFormatter(
            results_formatter_object,
            shutdown_formatter)
    else:
        return None


class EventBuilder(object):
    """Helper class to build test result event dictionaries."""

    BASE_DICTIONARY = None

    # Test Event Types
    TYPE_JOB_RESULT = "job_result"
    TYPE_TEST_RESULT = "test_result"
    TYPE_TEST_START = "test_start"
    TYPE_MARK_TEST_RERUN_ELIGIBLE = "test_eligible_for_rerun"

    RESULT_TYPES = set([
        TYPE_JOB_RESULT,
        TYPE_TEST_RESULT
        ])

    # Test/Job Status Tags
    STATUS_EXCEPTIONAL_EXIT = "exceptional_exit"
    STATUS_SUCCESS = "success"
    STATUS_FAILURE = "failure"
    STATUS_EXPECTED_FAILURE = "expected_failure"
    STATUS_EXPECTED_TIMEOUT = "expected_timeout"
    STATUS_UNEXPECTED_SUCCESS = "unexpected_success"
    STATUS_SKIP = "skip"
    STATUS_ERROR = "error"
    STATUS_TIMEOUT = "timeout"

    """Test methods or jobs with a status matching any of these
    status values will cause a testrun failure, unless
    the test methods rerun and do not trigger an issue when rerun."""
    TESTRUN_ERROR_STATUS_VALUES = set([
        STATUS_ERROR,
        STATUS_EXCEPTIONAL_EXIT,
        STATUS_FAILURE,
        STATUS_TIMEOUT
        ])

    @staticmethod
    def _get_test_name_info(test):
        """Returns (test-class-name, test-method-name) from a test case instance.

        @param test a unittest.TestCase instance.

        @return tuple containing (test class name, test method name)
        """
        test_class_components = test.id().split(".")
        test_class_name = ".".join(test_class_components[:-1])
        test_name = test_class_components[-1]
        return (test_class_name, test_name)

    @staticmethod
    def bare_event(event_type):
        """Creates an event with default additions, event type and timestamp.

        @param event_type the value set for the "event" key, used
        to distinguish events.

        @returns an event dictionary with all default additions, the "event"
        key set to the passed in event_type, and the event_time value set to
        time.time().
        """
        if EventBuilder.BASE_DICTIONARY is not None:
            # Start with a copy of the "always include" entries.
            event = dict(EventBuilder.BASE_DICTIONARY)
        else:
            event = {}

        event.update({
            "event": event_type,
            "event_time": time.time()
            })
        return event

    @staticmethod
    def _event_dictionary_common(test, event_type):
        """Returns an event dictionary setup with values for the given event type.

        @param test the unittest.TestCase instance

        @param event_type the name of the event type (string).

        @return event dictionary with common event fields set.
        """
        test_class_name, test_name = EventBuilder._get_test_name_info(test)

        event = EventBuilder.bare_event(event_type)
        event.update({
            "test_class": test_class_name,
            "test_name": test_name,
            "test_filename": inspect.getfile(test.__class__)
        })

        return event

    @staticmethod
    def _error_tuple_class(error_tuple):
        """Returns the unittest error tuple's error class as a string.

        @param error_tuple the error tuple provided by the test framework.

        @return the error type (typically an exception) raised by the
        test framework.
        """
        type_var = error_tuple[0]
        module = inspect.getmodule(type_var)
        if module:
            return "{}.{}".format(module.__name__, type_var.__name__)
        else:
            return type_var.__name__

    @staticmethod
    def _error_tuple_message(error_tuple):
        """Returns the unittest error tuple's error message.

        @param error_tuple the error tuple provided by the test framework.

        @return the error message provided by the test framework.
        """
        return str(error_tuple[1])

    @staticmethod
    def _error_tuple_traceback(error_tuple):
        """Returns the unittest error tuple's error message.

        @param error_tuple the error tuple provided by the test framework.

        @return the error message provided by the test framework.
        """
        return error_tuple[2]

    @staticmethod
    def _event_dictionary_test_result(test, status):
        """Returns an event dictionary with common test result fields set.

        @param test a unittest.TestCase instance.

        @param status the status/result of the test
        (e.g. "success", "failure", etc.)

        @return the event dictionary
        """
        event = EventBuilder._event_dictionary_common(
            test, EventBuilder.TYPE_TEST_RESULT)
        event["status"] = status
        return event

    @staticmethod
    def _event_dictionary_issue(test, status, error_tuple):
        """Returns an event dictionary with common issue-containing test result
        fields set.

        @param test a unittest.TestCase instance.

        @param status the status/result of the test
        (e.g. "success", "failure", etc.)

        @param error_tuple the error tuple as reported by the test runner.
        This is of the form (type<error>, error).

        @return the event dictionary
        """
        event = EventBuilder._event_dictionary_test_result(test, status)
        event["issue_class"] = EventBuilder._error_tuple_class(error_tuple)
        event["issue_message"] = EventBuilder._error_tuple_message(error_tuple)
        backtrace = EventBuilder._error_tuple_traceback(error_tuple)
        if backtrace is not None:
            event["issue_backtrace"] = traceback.format_tb(backtrace)
        return event

    @staticmethod
    def event_for_start(test):
        """Returns an event dictionary for the test start event.

        @param test a unittest.TestCase instance.

        @return the event dictionary
        """
        return EventBuilder._event_dictionary_common(
            test, EventBuilder.TYPE_TEST_START)

    @staticmethod
    def event_for_success(test):
        """Returns an event dictionary for a successful test.

        @param test a unittest.TestCase instance.

        @return the event dictionary
        """
        return EventBuilder._event_dictionary_test_result(
            test, EventBuilder.STATUS_SUCCESS)

    @staticmethod
    def event_for_unexpected_success(test, bugnumber):
        """Returns an event dictionary for a test that succeeded but was
        expected to fail.

        @param test a unittest.TestCase instance.

        @param bugnumber the issue identifier for the bug tracking the
        fix request for the test expected to fail (but is in fact
        passing here).

        @return the event dictionary

        """
        event = EventBuilder._event_dictionary_test_result(
            test, EventBuilder.STATUS_UNEXPECTED_SUCCESS)
        if bugnumber:
            event["bugnumber"] = str(bugnumber)
        return event

    @staticmethod
    def event_for_failure(test, error_tuple):
        """Returns an event dictionary for a test that failed.

        @param test a unittest.TestCase instance.

        @param error_tuple the error tuple as reported by the test runner.
        This is of the form (type<error>, error).

        @return the event dictionary
        """
        return EventBuilder._event_dictionary_issue(
            test, EventBuilder.STATUS_FAILURE, error_tuple)

    @staticmethod
    def event_for_expected_failure(test, error_tuple, bugnumber):
        """Returns an event dictionary for a test that failed as expected.

        @param test a unittest.TestCase instance.

        @param error_tuple the error tuple as reported by the test runner.
        This is of the form (type<error>, error).

        @param bugnumber the issue identifier for the bug tracking the
        fix request for the test expected to fail.

        @return the event dictionary

        """
        event = EventBuilder._event_dictionary_issue(
            test, EventBuilder.STATUS_EXPECTED_FAILURE, error_tuple)
        if bugnumber:
            event["bugnumber"] = str(bugnumber)
        return event

    @staticmethod
    def event_for_skip(test, reason):
        """Returns an event dictionary for a test that was skipped.

        @param test a unittest.TestCase instance.

        @param reason the reason why the test is being skipped.

        @return the event dictionary
        """
        event = EventBuilder._event_dictionary_test_result(
            test, EventBuilder.STATUS_SKIP)
        event["skip_reason"] = reason
        return event

    @staticmethod
    def event_for_error(test, error_tuple):
        """Returns an event dictionary for a test that hit a test execution error.

        @param test a unittest.TestCase instance.

        @param error_tuple the error tuple as reported by the test runner.
        This is of the form (type<error>, error).

        @return the event dictionary
        """
        return EventBuilder._event_dictionary_issue(
            test, EventBuilder.STATUS_ERROR, error_tuple)

    @staticmethod
    def event_for_cleanup_error(test, error_tuple):
        """Returns an event dictionary for a test that hit a test execution error
        during the test cleanup phase.

        @param test a unittest.TestCase instance.

        @param error_tuple the error tuple as reported by the test runner.
        This is of the form (type<error>, error).

        @return the event dictionary
        """
        event = EventBuilder._event_dictionary_issue(
            test, EventBuilder.STATUS_ERROR, error_tuple)
        event["issue_phase"] = "cleanup"
        return event

    @staticmethod
    def event_for_job_exceptional_exit(
            pid, worker_index, exception_code, exception_description,
            test_filename, command_line):
        """Creates an event for a job (i.e. process) exit due to signal.

        @param pid the process id for the job that failed
        @param worker_index optional id for the job queue running the process
        @param exception_code optional code
        (e.g. SIGTERM integer signal number)
        @param exception_description optional string containing symbolic
        representation of the issue (e.g. "SIGTERM")
        @param test_filename the path to the test filename that exited
        in some exceptional way.
        @param command_line the Popen-style list provided as the command line
        for the process that timed out.

        @return an event dictionary coding the job completion description.
        """
        event = EventBuilder.bare_event(EventBuilder.TYPE_JOB_RESULT)
        event["status"] = EventBuilder.STATUS_EXCEPTIONAL_EXIT
        if pid is not None:
            event["pid"] = pid
        if worker_index is not None:
            event["worker_index"] = int(worker_index)
        if exception_code is not None:
            event["exception_code"] = exception_code
        if exception_description is not None:
            event["exception_description"] = exception_description
        if test_filename is not None:
            event["test_filename"] = test_filename
        if command_line is not None:
            event["command_line"] = command_line
        return event

    @staticmethod
    def event_for_job_timeout(pid, worker_index, test_filename, command_line):
        """Creates an event for a job (i.e. process) timeout.

        @param pid the process id for the job that timed out
        @param worker_index optional id for the job queue running the process
        @param test_filename the path to the test filename that timed out.
        @param command_line the Popen-style list provided as the command line
        for the process that timed out.

        @return an event dictionary coding the job completion description.
        """
        event = EventBuilder.bare_event(EventBuilder.TYPE_JOB_RESULT)
        event["status"] = "timeout"
        if pid is not None:
            event["pid"] = pid
        if worker_index is not None:
            event["worker_index"] = int(worker_index)
        if test_filename is not None:
            event["test_filename"] = test_filename
        if command_line is not None:
            event["command_line"] = command_line
        return event

    @staticmethod
    def event_for_mark_test_rerun_eligible(test):
        """Creates an event that indicates the specified test is explicitly
        eligible for rerun.

        Note there is a mode that will enable test rerun eligibility at the
        global level.  These markings for explicit rerun eligibility are
        intended for the mode of running where only explicitly rerunnable
        tests are rerun upon hitting an issue.

        @param test the TestCase instance to which this pertains.

        @return an event that specifies the given test as being eligible to
        be rerun.
        """
        event = EventBuilder._event_dictionary_common(
            test,
            EventBuilder.TYPE_MARK_TEST_RERUN_ELIGIBLE)
        return event

    @staticmethod
    def add_entries_to_all_events(entries_dict):
        """Specifies a dictionary of entries to add to all test events.

        This provides a mechanism for, say, a parallel test runner to
        indicate to each inferior dotest.py that it should add a
        worker index to each.

        Calling this method replaces all previous entries added
        by a prior call to this.

        Event build methods will overwrite any entries that collide.
        Thus, the passed in dictionary is the base, which gets merged
        over by event building when keys collide.

        @param entries_dict a dictionary containing key and value
        pairs that should be merged into all events created by the
        event generator.  May be None to clear out any extra entries.
        """
        EventBuilder.BASE_DICTIONARY = dict(entries_dict)


class ResultsFormatter(object):
    """Provides interface to formatting test results out to a file-like object.

    This class allows the LLDB test framework's raw test-realted
    events to be processed and formatted in any manner desired.
    Test events are represented by python dictionaries, formatted
    as in the EventBuilder class above.

    ResultFormatter instances are given a file-like object in which
    to write their results.

    ResultFormatter lifetime looks like the following:

    # The result formatter is created.
    # The argparse options dictionary is generated from calling
    # the SomeResultFormatter.arg_parser() with the options data
    # passed to dotest.py via the "--results-formatter-options"
    # argument.  See the help on that for syntactic requirements
    # on getting that parsed correctly.
    formatter = SomeResultFormatter(file_like_object, argpared_options_dict)

    # Single call to session start, before parsing any events.
    formatter.begin_session()

    formatter.handle_event({"event":"initialize",...})

    # Zero or more calls specified for events recorded during the test session.
    # The parallel test runner manages getting results from all the inferior
    # dotest processes, so from a new format perspective, don't worry about
    # that.  The formatter will be presented with a single stream of events
    # sandwiched between a single begin_session()/end_session() pair in the
    # parallel test runner process/thread.
    for event in zero_or_more_test_events():
        formatter.handle_event(event)

    # Single call to terminate/wrap-up. Formatters that need all the
    # data before they can print a correct result (e.g. xUnit/JUnit),
    # this is where the final report can be generated.
    formatter.handle_event({"event":"terminate",...})

    It is not the formatter's responsibility to close the file_like_object.
    (i.e. do not close it).

    The lldb test framework passes these test events in real time, so they
    arrive as they come in.

    In the case of the parallel test runner, the dotest inferiors
    add a 'pid' field to the dictionary that indicates which inferior
    pid generated the event.

    Note more events may be added in the future to support richer test
    reporting functionality. One example: creating a true flaky test
    result category so that unexpected successes really mean the test
    is marked incorrectly (either should be marked flaky, or is indeed
    passing consistently now and should have the xfail marker
    removed). In this case, a flaky_success and flaky_fail event
    likely will be added to capture these and support reporting things
    like percentages of flaky test passing so we can see if we're
    making some things worse/better with regards to failure rates.

    Another example: announcing all the test methods that are planned
    to be run, so we can better support redo operations of various kinds
    (redo all non-run tests, redo non-run tests except the one that
    was running [perhaps crashed], etc.)

    Implementers are expected to override all the public methods
    provided in this class. See each method's docstring to see
    expectations about when the call should be chained.

    """
    @classmethod
    def arg_parser(cls):
        """@return arg parser used to parse formatter-specific options."""
        parser = argparse.ArgumentParser(
            description='{} options'.format(cls.__name__),
            usage=('dotest.py --results-formatter-options='
                   '"--option1 value1 [--option2 value2 [...]]"'))
        return parser

    def __init__(self, out_file, options):
        super(ResultsFormatter, self).__init__()
        self.out_file = out_file
        self.options = options
        self.using_terminal = False
        if not self.out_file:
            raise Exception("ResultsFormatter created with no file object")
        self.start_time_by_test = {}
        self.terminate_called = False

        # Store counts of test_result events by status.
        self.result_status_counts = {
            EventBuilder.STATUS_SUCCESS: 0,
            EventBuilder.STATUS_EXPECTED_FAILURE: 0,
            EventBuilder.STATUS_EXPECTED_TIMEOUT: 0,
            EventBuilder.STATUS_SKIP: 0,
            EventBuilder.STATUS_UNEXPECTED_SUCCESS: 0,
            EventBuilder.STATUS_FAILURE: 0,
            EventBuilder.STATUS_ERROR: 0,
            EventBuilder.STATUS_TIMEOUT: 0,
            EventBuilder.STATUS_EXCEPTIONAL_EXIT: 0
        }

        # Track the most recent test start event by worker index.
        # We'll use this to assign TIMEOUT and exceptional
        # exits to the most recent test started on a given
        # worker index.
        self.started_tests_by_worker = {}

        # Lock that we use while mutating inner state, like the
        # total test count and the elements.  We minimize how
        # long we hold the lock just to keep inner state safe, not
        # entirely consistent from the outside.
        self.lock = threading.Lock()

        # Keeps track of the test base filenames for tests that
        # are expected to timeout.  If a timeout occurs in any test
        # basename that matches this list, that result should be
        # converted into a non-issue.  We'll create an expected
        # timeout test status for this.
        self.expected_timeouts_by_basename = set()

        # Keep track of rerun-eligible tests.
        # This is a set that contains tests saved as:
        # {test_filename}:{test_class}:{test_name}
        self.rerun_eligible_tests = set()

        # A dictionary of test files that had a failing
        # test, in the format of:
        # key = test path, value = array of test methods that need rerun
        self.tests_for_rerun = {}

    @classmethod
    def _make_rerun_eligibility_key(cls, test_result_event):
        if test_result_event is None:
            return None
        component_count = 0
        if "test_filename" in test_result_event:
            key = test_result_event["test_filename"]
            component_count += 1
        if "test_class" in test_result_event:
            if component_count > 0:
                key += "."
            key += test_result_event["test_class"]
            component_count += 1
        if "test_name" in test_result_event:
            if component_count > 0:
                key += "."
            key += test_result_event["test_name"]
            component_count += 1
        return key

    def _mark_test_for_rerun_eligibility(self, test_result_event):
        key = self._make_rerun_eligibility_key(test_result_event)
        if key is not None:
            self.rerun_eligible_tests.add(key)
        else:
            sys.stderr.write(
                "\nerror: test marked for re-run eligibility but "
                "failed to create key.\n")

    def _maybe_add_test_to_rerun_list(self, result_event):
        key = self._make_rerun_eligibility_key(result_event)
        if key is not None:
            if key in self.rerun_eligible_tests or configuration.rerun_all_issues:
                test_filename = result_event.get("test_filename", None)
                if test_filename is not None:
                    test_name = result_event.get("test_name", None)
                    if not test_filename in self.tests_for_rerun:
                        self.tests_for_rerun[test_filename] = []
                    if test_name is not None:
                        self.tests_for_rerun[test_filename].append(test_name)
        else:
            sys.stderr.write(
                "\nerror: couldn't add testrun-failing test to rerun "
                "list because no eligibility key could be created.\n")

    def _maybe_remap_job_result_event(self, test_event):
        """Remaps timeout/exceptional exit job results to last test method running.

        @param test_event the job_result test event.  This is an in/out
        parameter.  It will be modified if it can be mapped to a test_result
        of the same status, using details from the last-running test method
        known to be most recently started on the same worker index.
        """
        test_start = None

        job_status = test_event["status"]
        if job_status in [
                EventBuilder.STATUS_TIMEOUT,
                EventBuilder.STATUS_EXCEPTIONAL_EXIT]:
            worker_index = test_event.get("worker_index", None)
            if worker_index is not None:
                test_start = self.started_tests_by_worker.get(
                    worker_index, None)

        # If we have a test start to remap, do it here.
        if test_start is not None:
            test_event["event"] = EventBuilder.TYPE_TEST_RESULT

            # Fill in all fields from test start not present in
            # job status message.
            for (start_key, start_value) in test_start.items():
                if start_key not in test_event:
                    test_event[start_key] = start_value

    def _maybe_remap_expected_timeout(self, event):
        if event is None:
            return

        status = event.get("status", None)
        if status is None or status != EventBuilder.STATUS_TIMEOUT:
            return

        # Check if the timeout test's basename is in the expected timeout
        # list.  If so, convert to an expected timeout.
        basename = os.path.basename(event.get("test_filename", ""))
        if basename in self.expected_timeouts_by_basename:
            # Convert to an expected timeout.
            event["status"] = EventBuilder.STATUS_EXPECTED_TIMEOUT

    def handle_event(self, test_event):
        """Handles the test event for collection into the formatter output.

        Derived classes may override this but should call down to this
        implementation first.

        @param test_event the test event as formatted by one of the
        event_for_* calls.
        """
        # Keep track of whether terminate was received.  We do this so
        # that a process can call the 'terminate' event on its own, to
        # close down a formatter at the appropriate time.  Then the
        # atexit() cleanup can call the "terminate if it hasn't been
        # called yet".
        if test_event is not None:
            event_type = test_event.get("event", "")
            # We intentionally allow event_type to be checked anew
            # after this check below since this check may rewrite
            # the event type
            if event_type == EventBuilder.TYPE_JOB_RESULT:
                # Possibly convert the job status (timeout, exceptional exit)
                # to an appropriate test_result event.
                self._maybe_remap_job_result_event(test_event)
                event_type = test_event.get("event", "")

            # Remap timeouts to expected timeouts.
            if event_type in EventBuilder.RESULT_TYPES:
                self._maybe_remap_expected_timeout(test_event)
                event_type = test_event.get("event", "")

            if event_type == "terminate":
                self.terminate_called = True
            elif event_type in EventBuilder.RESULT_TYPES:
                # Keep track of event counts per test/job result status type.
                # The only job (i.e. inferior process) results that make it
                # here are ones that cannot be remapped to the most recently
                # started test for the given worker index.
                status = test_event["status"]
                self.result_status_counts[status] += 1
                # Clear the most recently started test for the related worker.
                worker_index = test_event.get("worker_index", None)
                if worker_index is not None:
                    self.started_tests_by_worker.pop(worker_index, None)

                if status in EventBuilder.TESTRUN_ERROR_STATUS_VALUES:
                    # A test/job status value in any of those status values
                    # causes a testrun failure.  If such a test fails, check
                    # whether it can be rerun.  If it can be rerun, add it
                    # to the rerun job.
                    self._maybe_add_test_to_rerun_list(test_event)
            elif event_type == EventBuilder.TYPE_TEST_START:
                # Keep track of the most recent test start event
                # for the related worker.
                worker_index = test_event.get("worker_index", None)
                if worker_index is not None:
                    self.started_tests_by_worker[worker_index] = test_event
            elif event_type == EventBuilder.TYPE_MARK_TEST_RERUN_ELIGIBLE:
                self._mark_test_for_rerun_eligibility(test_event)

    def set_expected_timeouts_by_basename(self, basenames):
        """Specifies a list of test file basenames that are allowed to timeout
        without being called out as a timeout issue.

        These fall into a new status category called STATUS_EXPECTED_TIMEOUT.
        """
        if basenames is not None:
            for basename in basenames:
                self.expected_timeouts_by_basename.add(basename)

    def track_start_time(self, test_class, test_name, start_time):
        """tracks the start time of a test so elapsed time can be computed.

        this alleviates the need for test results to be processed serially
        by test.  it will save the start time for the test so that
        elapsed_time_for_test() can compute the elapsed time properly.
        """
        if test_class is None or test_name is None:
            return

        test_key = "{}.{}".format(test_class, test_name)
        with self.lock:
            self.start_time_by_test[test_key] = start_time

    def elapsed_time_for_test(self, test_class, test_name, end_time):
        """returns the elapsed time for a test.

        this function can only be called once per test and requires that
        the track_start_time() method be called sometime prior to calling
        this method.
        """
        if test_class is None or test_name is None:
            return -2.0

        test_key = "{}.{}".format(test_class, test_name)
        with self.lock:
            if test_key not in self.start_time_by_test:
                return -1.0
            else:
                start_time = self.start_time_by_test[test_key]
            del self.start_time_by_test[test_key]
        return end_time - start_time

    def is_using_terminal(self):
        """returns true if this results formatter is using the terminal and
        output should be avoided."""
        return self.using_terminal

    def send_terminate_as_needed(self):
        """sends the terminate event if it hasn't been received yet."""
        if not self.terminate_called:
            terminate_event = EventBuilder.bare_event("terminate")
            self.handle_event(terminate_event)

    # Derived classes may require self access
    # pylint: disable=no-self-use
    def replaces_summary(self):
        """Returns whether the results formatter includes a summary
        suitable to replace the old lldb test run results.

        @return True if the lldb test runner can skip its summary
        generation when using this results formatter; False otherwise.
        """
        return False

    def counts_by_test_result_status(self, status):
        """Returns number of test method results for the given status.

        @status_result a test result status (e.g. success, fail, skip)
        as defined by the EventBuilder.STATUS_* class members.

        @return an integer returning the number of test methods matching
        the given test result status.
        """
        return self.result_status_counts[status]


class RawPickledFormatter(ResultsFormatter):
    """Formats events as a pickled stream.

    The parallel test runner has inferiors pickle their results and send them
    over a socket back to the parallel test.  The parallel test runner then
    aggregates them into the final results formatter (e.g. xUnit).
    """

    @classmethod
    def arg_parser(cls):
        """@return arg parser used to parse formatter-specific options."""
        parser = super(RawPickledFormatter, cls).arg_parser()
        return parser

    def __init__(self, out_file, options):
        super(RawPickledFormatter, self).__init__(out_file, options)
        self.pid = os.getpid()

    def handle_event(self, test_event):
        super(RawPickledFormatter, self).handle_event(test_event)

        # Convert initialize/terminate events into job_begin/job_end events.
        event_type = test_event["event"]
        if event_type is None:
            return

        if event_type == "initialize":
            test_event["event"] = "job_begin"
        elif event_type == "terminate":
            test_event["event"] = "job_end"

        # Tack on the pid.
        test_event["pid"] = self.pid

        # Send it as {serialized_length_of_serialized_bytes}{serialized_bytes}
        import struct
        msg = cPickle.dumps(test_event)
        packet = struct.pack("!I%ds" % len(msg), len(msg), msg)
        self.out_file.send(packet)


class DumpFormatter(ResultsFormatter):
    """Formats events to the file as their raw python dictionary format."""

    def handle_event(self, test_event):
        super(DumpFormatter, self).handle_event(test_event)
        self.out_file.write("\n" + pprint.pformat(test_event) + "\n")

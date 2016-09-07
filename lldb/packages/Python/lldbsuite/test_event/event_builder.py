"""
    The LLVM Compiler Infrastructure

This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.

Provides a class to build Python test event data structures.
"""

from __future__ import print_function
from __future__ import absolute_import

# System modules
import inspect
import time
import traceback

# Third-party modules

# LLDB modules
from . import build_exception


class EventBuilder(object):
    """Helper class to build test result event dictionaries."""

    BASE_DICTIONARY = None

    # Test Event Types
    TYPE_JOB_RESULT = "job_result"
    TYPE_TEST_RESULT = "test_result"
    TYPE_TEST_START = "test_start"
    TYPE_MARK_TEST_RERUN_ELIGIBLE = "test_eligible_for_rerun"
    TYPE_MARK_TEST_EXPECTED_FAILURE = "test_expected_failure"
    TYPE_SESSION_TERMINATE = "terminate"

    RESULT_TYPES = {TYPE_JOB_RESULT, TYPE_TEST_RESULT}

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
    TESTRUN_ERROR_STATUS_VALUES = {
        STATUS_ERROR,
        STATUS_EXCEPTIONAL_EXIT,
        STATUS_FAILURE,
        STATUS_TIMEOUT}

    @staticmethod
    def _get_test_name_info(test):
        """Returns (test-class-name, test-method-name) from a test case instance.

        @param test a unittest.TestCase instance.

        @return tuple containing (test class name, test method name)
        """
        test_class_components = test.id().split(".")
        test_class_name = ".".join(test_class_components[:-1])
        test_name = test_class_components[-1]
        return test_class_name, test_name

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
    def _assert_is_python_sourcefile(test_filename):
        if test_filename is not None:
            if not test_filename.endswith(".py"):
                raise Exception(
                    "source python filename has unexpected extension: {}".format(test_filename))
        return test_filename

    @staticmethod
    def _event_dictionary_common(test, event_type):
        """Returns an event dictionary setup with values for the given event type.

        @param test the unittest.TestCase instance

        @param event_type the name of the event type (string).

        @return event dictionary with common event fields set.
        """
        test_class_name, test_name = EventBuilder._get_test_name_info(test)

        # Determine the filename for the test case.  If there is an attribute
        # for it, use it.  Otherwise, determine from the TestCase class path.
        if hasattr(test, "test_filename"):
            test_filename = EventBuilder._assert_is_python_sourcefile(
                test.test_filename)
        else:
            test_filename = EventBuilder._assert_is_python_sourcefile(
                inspect.getsourcefile(test.__class__))

        event = EventBuilder.bare_event(event_type)
        event.update({
            "test_class": test_class_name,
            "test_name": test_name,
            "test_filename": test_filename
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
        event = EventBuilder._event_dictionary_issue(
            test, EventBuilder.STATUS_ERROR, error_tuple)
        event["issue_phase"] = "test"
        return event

    @staticmethod
    def event_for_build_error(test, error_tuple):
        """Returns an event dictionary for a test that hit a test execution error
        during the test cleanup phase.

        @param test a unittest.TestCase instance.

        @param error_tuple the error tuple as reported by the test runner.
        This is of the form (type<error>, error).

        @return the event dictionary
        """
        event = EventBuilder._event_dictionary_issue(
            test, EventBuilder.STATUS_ERROR, error_tuple)
        event["issue_phase"] = "build"

        build_error = error_tuple[1]
        event["build_command"] = build_error.command
        event["build_error"] = build_error.build_error
        return event

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
    def event_for_job_test_add_error(test_filename, exception, backtrace):
        event = EventBuilder.bare_event(EventBuilder.TYPE_JOB_RESULT)
        event["status"] = EventBuilder.STATUS_ERROR
        if test_filename is not None:
            event["test_filename"] = EventBuilder._assert_is_python_sourcefile(
                test_filename)
        if exception is not None and "__class__" in dir(exception):
            event["issue_class"] = exception.__class__
        event["issue_message"] = exception
        if backtrace is not None:
            event["issue_backtrace"] = backtrace
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
        @param command_line the Popen()-style list provided as the command line
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
            event["test_filename"] = EventBuilder._assert_is_python_sourcefile(
                test_filename)
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
            event["test_filename"] = EventBuilder._assert_is_python_sourcefile(
                test_filename)
        if command_line is not None:
            event["command_line"] = command_line
        return event

    @staticmethod
    def event_for_mark_test_rerun_eligible(test):
        """Creates an event that indicates the specified test is explicitly
        eligible for rerun.

        Note there is a mode that will enable test rerun eligibility at the
        global level.  These markings for explicit rerun eligibility are
        intended for the mode of running where only explicitly re-runnable
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
    def event_for_mark_test_expected_failure(test):
        """Creates an event that indicates the specified test is expected
        to fail.

        @param test the TestCase instance to which this pertains.

        @return an event that specifies the given test is expected to fail.
        """
        event = EventBuilder._event_dictionary_common(
            test,
            EventBuilder.TYPE_MARK_TEST_EXPECTED_FAILURE)
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

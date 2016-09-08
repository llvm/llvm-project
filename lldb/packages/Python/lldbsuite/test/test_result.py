"""
                     The LLVM Compiler Infrastructure

This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.

Provides the LLDBTestResult class, which holds information about progress
and results of a single test run.
"""

from __future__ import absolute_import
from __future__ import print_function

# System modules
import inspect
import os

# Third-party modules
import unittest2

# LLDB Modules
from . import configuration
from lldbsuite.test_event.event_builder import EventBuilder
from lldbsuite.test_event import build_exception


class LLDBTestResult(unittest2.TextTestResult):
    """
    Enforce a singleton pattern to allow introspection of test progress.

    Overwrite addError(), addFailure(), and addExpectedFailure() methods
    to enable each test instance to track its failure/error status.  It
    is used in the LLDB test framework to emit detailed trace messages
    to a log file for easier human inspection of test failures/errors.
    """
    __singleton__ = None
    __ignore_singleton__ = False

    @staticmethod
    def getTerminalSize():
        import os
        env = os.environ

        def ioctl_GWINSZ(fd):
            try:
                import fcntl
                import termios
                import struct
                import os
                cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
                                                     '1234'))
            except:
                return
            return cr
        cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
        if not cr:
            try:
                fd = os.open(os.ctermid(), os.O_RDONLY)
                cr = ioctl_GWINSZ(fd)
                os.close(fd)
            except:
                pass
        if not cr:
            cr = (env.get('LINES', 25), env.get('COLUMNS', 80))
        return int(cr[1]), int(cr[0])

    def __init__(self, *args):
        if not LLDBTestResult.__ignore_singleton__ and LLDBTestResult.__singleton__:
            raise Exception("LLDBTestResult instantiated more than once")
        super(LLDBTestResult, self).__init__(*args)
        LLDBTestResult.__singleton__ = self
        # Now put this singleton into the lldb module namespace.
        configuration.test_result = self
        # Computes the format string for displaying the counter.
        counterWidth = len(str(configuration.suite.countTestCases()))
        self.fmt = "%" + str(counterWidth) + "d: "
        self.indentation = ' ' * (counterWidth + 2)
        # This counts from 1 .. suite.countTestCases().
        self.counter = 0
        (width, height) = LLDBTestResult.getTerminalSize()
        self.results_formatter = configuration.results_formatter_object

    def _config_string(self, test):
        compiler = getattr(test, "getCompiler", None)
        arch = getattr(test, "getArchitecture", None)
        return "%s-%s" % (compiler() if compiler else "",
                          arch() if arch else "")

    def _exc_info_to_string(self, err, test):
        """Overrides superclass TestResult's method in order to append
        our test config info string to the exception info string."""
        if hasattr(test, "getArchitecture") and hasattr(test, "getCompiler"):
            return '%sConfig=%s-%s' % (super(LLDBTestResult,
                                             self)._exc_info_to_string(err,
                                                                       test),
                                       test.getArchitecture(),
                                       test.getCompiler())
        else:
            return super(LLDBTestResult, self)._exc_info_to_string(err, test)

    def getDescription(self, test):
        doc_first_line = test.shortDescription()
        if self.descriptions and doc_first_line:
            return '\n'.join((str(test), self.indentation + doc_first_line))
        else:
            return str(test)

    def getCategoriesForTest(self, test):
        """
        Gets all the categories for the currently running test method in test case
        """
        test_categories = []
        test_method = getattr(test, test._testMethodName)
        if test_method is not None and hasattr(test_method, "categories"):
            test_categories.extend(test_method.categories)

        test_categories.extend(test.getCategories())

        return test_categories

    def hardMarkAsSkipped(self, test):
        getattr(test, test._testMethodName).__func__.__unittest_skip__ = True
        getattr(
            test,
            test._testMethodName).__func__.__unittest_skip_why__ = "test case does not fall in any category of interest for this run"
        test.__class__.__unittest_skip__ = True
        test.__class__.__unittest_skip_why__ = "test case does not fall in any category of interest for this run"

    def startTest(self, test):
        if configuration.shouldSkipBecauseOfCategories(
                self.getCategoriesForTest(test)):
            self.hardMarkAsSkipped(test)
        configuration.setCrashInfoHook(
            "%s at %s" %
            (str(test), inspect.getfile(
                test.__class__)))
        self.counter += 1
        # if self.counter == 4:
        #    import crashinfo
        #    crashinfo.testCrashReporterDescription(None)
        test.test_number = self.counter
        if self.showAll:
            self.stream.write(self.fmt % self.counter)
        super(LLDBTestResult, self).startTest(test)
        if self.results_formatter:
            self.results_formatter.handle_event(
                EventBuilder.event_for_start(test))

    def addSuccess(self, test):
        super(LLDBTestResult, self).addSuccess(test)
        if configuration.parsable:
            self.stream.write(
                "PASS: LLDB (%s) :: %s\n" %
                (self._config_string(test), str(test)))
        if self.results_formatter:
            self.results_formatter.handle_event(
                EventBuilder.event_for_success(test))

    def _isBuildError(self, err_tuple):
        exception = err_tuple[1]
        return isinstance(exception, build_exception.BuildError)

    def _getTestPath(self, test):
        if test is None:
            return ""
        elif hasattr(test, "test_filename"):
            return test.test_filename
        else:
            return inspect.getsourcefile(test.__class__)

    def _saveBuildErrorTuple(self, test, err):
        # Adjust the error description so it prints the build command and build error
        # rather than an uninformative Python backtrace.
        build_error = err[1]
        error_description = "{}\nTest Directory:\n{}".format(
            str(build_error),
            os.path.dirname(self._getTestPath(test)))
        self.errors.append((test, error_description))
        self._mirrorOutput = True

    def addError(self, test, err):
        configuration.sdir_has_content = True
        if self._isBuildError(err):
            self._saveBuildErrorTuple(test, err)
        else:
            super(LLDBTestResult, self).addError(test, err)

        method = getattr(test, "markError", None)
        if method:
            method()
        if configuration.parsable:
            self.stream.write(
                "FAIL: LLDB (%s) :: %s\n" %
                (self._config_string(test), str(test)))
        if self.results_formatter:
            # Handle build errors as a separate event type
            if self._isBuildError(err):
                error_event = EventBuilder.event_for_build_error(test, err)
            else:
                error_event = EventBuilder.event_for_error(test, err)
            self.results_formatter.handle_event(error_event)

    def addCleanupError(self, test, err):
        configuration.sdir_has_content = True
        super(LLDBTestResult, self).addCleanupError(test, err)
        method = getattr(test, "markCleanupError", None)
        if method:
            method()
        if configuration.parsable:
            self.stream.write(
                "CLEANUP ERROR: LLDB (%s) :: %s\n" %
                (self._config_string(test), str(test)))
        if self.results_formatter:
            self.results_formatter.handle_event(
                EventBuilder.event_for_cleanup_error(
                    test, err))

    def addFailure(self, test, err):
        configuration.sdir_has_content = True
        super(LLDBTestResult, self).addFailure(test, err)
        method = getattr(test, "markFailure", None)
        if method:
            method()
        if configuration.parsable:
            self.stream.write(
                "FAIL: LLDB (%s) :: %s\n" %
                (self._config_string(test), str(test)))
        if configuration.useCategories:
            test_categories = self.getCategoriesForTest(test)
            for category in test_categories:
                if category in configuration.failuresPerCategory:
                    configuration.failuresPerCategory[
                        category] = configuration.failuresPerCategory[category] + 1
                else:
                    configuration.failuresPerCategory[category] = 1
        if self.results_formatter:
            self.results_formatter.handle_event(
                EventBuilder.event_for_failure(test, err))

    def addExpectedFailure(self, test, err, bugnumber):
        configuration.sdir_has_content = True
        super(LLDBTestResult, self).addExpectedFailure(test, err, bugnumber)
        method = getattr(test, "markExpectedFailure", None)
        if method:
            method(err, bugnumber)
        if configuration.parsable:
            self.stream.write(
                "XFAIL: LLDB (%s) :: %s\n" %
                (self._config_string(test), str(test)))
        if self.results_formatter:
            self.results_formatter.handle_event(
                EventBuilder.event_for_expected_failure(
                    test, err, bugnumber))

    def addSkip(self, test, reason):
        configuration.sdir_has_content = True
        super(LLDBTestResult, self).addSkip(test, reason)
        method = getattr(test, "markSkippedTest", None)
        if method:
            method()
        if configuration.parsable:
            self.stream.write(
                "UNSUPPORTED: LLDB (%s) :: %s (%s) \n" %
                (self._config_string(test), str(test), reason))
        if self.results_formatter:
            self.results_formatter.handle_event(
                EventBuilder.event_for_skip(test, reason))

    def addUnexpectedSuccess(self, test, bugnumber):
        configuration.sdir_has_content = True
        super(LLDBTestResult, self).addUnexpectedSuccess(test, bugnumber)
        method = getattr(test, "markUnexpectedSuccess", None)
        if method:
            method(bugnumber)
        if configuration.parsable:
            self.stream.write(
                "XPASS: LLDB (%s) :: %s\n" %
                (self._config_string(test), str(test)))
        if self.results_formatter:
            self.results_formatter.handle_event(
                EventBuilder.event_for_unexpected_success(
                    test, bugnumber))

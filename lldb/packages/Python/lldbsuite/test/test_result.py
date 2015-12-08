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
import platform
import subprocess


# Third-party modules
import unittest2

# LLDB Modules
import lldbsuite
from . import configuration
from .result_formatter import EventBuilder


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
                import fcntl, termios, struct, os
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
        self.progressbar = None
        if width > 10 and not configuration.parsable and configuration.progress_bar:
            try:
                self.progressbar = progress.ProgressWithEvents(
                    stdout=self.stream,
                    start=0,
                    end=configuration.suite.countTestCases(),
                    width=width-10)
            except:
                self.progressbar = None
        self.results_formatter = configuration.results_formatter_object

    def _config_string(self, test):
        compiler = getattr(test, "getCompiler", None)
        arch = getattr(test, "getArchitecture", None)
        return "%s-%s" % (compiler() if compiler else "", arch() if arch else "")

    def _exc_info_to_string(self, err, test):
        """Overrides superclass TestResult's method in order to append
        our test config info string to the exception info string."""
        if hasattr(test, "getArchitecture") and hasattr(test, "getCompiler"):
            return '%sConfig=%s-%s' % (super(LLDBTestResult, self)._exc_info_to_string(err, test),
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

    def getCategoriesForTest(self,test):
        if hasattr(test,"_testMethodName"):
            test_method = getattr(test,"_testMethodName")
            test_method = getattr(test,test_method)
        else:
            test_method = None
        if test_method != None and hasattr(test_method,"getCategories"):
            test_categories = test_method.getCategories(test)
        elif hasattr(test,"getCategories"):
            test_categories = test.getCategories()
        elif inspect.ismethod(test) and test.__self__ != None and hasattr(test.__self__,"getCategories"):
            test_categories = test.__self__.getCategories()
        else:
            test_categories = []
        if test_categories == None:
            test_categories = []
        return test_categories

    def hardMarkAsSkipped(self,test):
        getattr(test, test._testMethodName).__func__.__unittest_skip__ = True
        getattr(test, test._testMethodName).__func__.__unittest_skip_why__ = "test case does not fall in any category of interest for this run"
        test.__class__.__unittest_skip__ = True
        test.__class__.__unittest_skip_why__ = "test case does not fall in any category of interest for this run"

    def startTest(self, test):
        if configuration.shouldSkipBecauseOfCategories(self.getCategoriesForTest(test)):
            self.hardMarkAsSkipped(test)
        configuration.setCrashInfoHook("%s at %s" % (str(test),inspect.getfile(test.__class__)))
        self.counter += 1
        #if self.counter == 4:
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
            self.stream.write("PASS: LLDB (%s) :: %s\n" % (self._config_string(test), str(test)))
        if self.results_formatter:
            self.results_formatter.handle_event(
                EventBuilder.event_for_success(test))

    def addError(self, test, err):
        configuration.sdir_has_content = True
        super(LLDBTestResult, self).addError(test, err)
        method = getattr(test, "markError", None)
        if method:
            method()
        if configuration.parsable:
            self.stream.write("FAIL: LLDB (%s) :: %s\n" % (self._config_string(test), str(test)))
        if self.results_formatter:
            self.results_formatter.handle_event(
                EventBuilder.event_for_error(test, err))

    def addCleanupError(self, test, err):
        configuration.sdir_has_content = True
        super(LLDBTestResult, self).addCleanupError(test, err)
        method = getattr(test, "markCleanupError", None)
        if method:
            method()
        if configuration.parsable:
            self.stream.write("CLEANUP ERROR: LLDB (%s) :: %s\n" % (self._config_string(test), str(test)))
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
            self.stream.write("FAIL: LLDB (%s) :: %s\n" % (self._config_string(test), str(test)))
        if configuration.useCategories:
            test_categories = self.getCategoriesForTest(test)
            for category in test_categories:
                if category in configuration.failuresPerCategory:
                    configuration.failuresPerCategory[category] = configuration.failuresPerCategory[category] + 1
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
            self.stream.write("XFAIL: LLDB (%s) :: %s\n" % (self._config_string(test), str(test)))
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
            self.stream.write("UNSUPPORTED: LLDB (%s) :: %s (%s) \n" % (self._config_string(test), str(test), reason))
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
            self.stream.write("XPASS: LLDB (%s) :: %s\n" % (self._config_string(test), str(test)))
        if self.results_formatter:
            self.results_formatter.handle_event(
                EventBuilder.event_for_unexpected_success(
                    test, bugnumber))

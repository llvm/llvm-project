from __future__ import absolute_import
import os

import subprocess
import sys

import lit.Test
import lit.TestRunner
import lit.util
from lit.formats.base import TestFormat


class LLDBTest(TestFormat):
    def __init__(self, dotest_cmd):
        self.dotest_cmd = dotest_cmd

    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig,
                            localConfig):
        source_path = testSuite.getSourcePath(path_in_suite)
        for filename in os.listdir(source_path):
            # Ignore dot files and excluded tests.
            if (filename.startswith('.') or filename in localConfig.excludes):
                continue

            # Ignore files that don't start with 'Test'.
            if not filename.startswith('Test'):
                continue

            filepath = os.path.join(source_path, filename)
            if not os.path.isdir(filepath):
                base, ext = os.path.splitext(filename)
                if ext in localConfig.suffixes:
                    yield lit.Test.Test(testSuite, path_in_suite +
                                        (filename, ), localConfig)

    def execute(self, test, litConfig):
        if litConfig.noExecute:
            return lit.Test.PASS, ''

        if test.config.lldb_disable_python:
            return (lit.Test.UNSUPPORTED, 'Python module disabled')

        if test.config.unsupported:
            return (lit.Test.UNSUPPORTED, 'Test is unsupported')

        testPath, testFile = os.path.split(test.getSourcePath())
        # On Windows, the system does not always correctly interpret shebang lines.
        # To make sure we can execute the tests, add python exe as the first parameter
        # of the command.
        cmd = [sys.executable] + self.dotest_cmd + [testPath, '-p', testFile]

        try:
            out, err, exitCode = lit.util.executeCommand(
                cmd,
                env=test.config.environment,
                timeout=litConfig.maxIndividualTestTime)
        except lit.util.ExecuteCommandTimeoutException:
            return (lit.Test.TIMEOUT, 'Reached timeout of {} seconds'.format(
                litConfig.maxIndividualTestTime))

        if exitCode:
            return lit.Test.FAIL, out + err

        passing_test_line = 'RESULT: PASSED'
        if passing_test_line not in out and passing_test_line not in err:
            msg = ('Unable to find %r in dotest output:\n\n%s%s' %
                   (passing_test_line, out, err))
            return lit.Test.UNRESOLVED, msg

        return lit.Test.PASS, ''

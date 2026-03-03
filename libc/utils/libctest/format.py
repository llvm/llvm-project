# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

"""
Lit test format for LLVM libc tests.

This format discovers pre-built test executables in the build directory
and runs them. It extends lit's ExecutableTest format.

The lit config sets test_source_root == test_exec_root (both to the build
directory), following the pattern used by llvm/test/Unit/lit.cfg.py.

Test executables are discovered by _isTestExecutable() and run by execute().
"""

import os
import shlex

import lit.formats
import lit.Test
import lit.util


class LibcTest(lit.formats.ExecutableTest):
    """
    Test format for libc unit tests.

    Extends ExecutableTest to discover pre-built test executables in the
    build directory rather than the source directory.
    """

    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig, localConfig):
        """
        Discover test executables in the build directory.

        Since test_source_root == test_exec_root (both point to build dir),
        we use getSourcePath() to find test executables.
        """
        source_path = testSuite.getSourcePath(path_in_suite)

        # Look for test executables in the build directory
        if not os.path.isdir(source_path):
            return

        # Sort for deterministic test discovery/output ordering.
        for filename in sorted(os.listdir(source_path)):
            filepath = os.path.join(source_path, filename)

            # Match our test executable pattern
            if self._isTestExecutable(filename, filepath):
                # Create a test with the executable name
                yield lit.Test.Test(testSuite, path_in_suite + (filename,), localConfig)

    def _isTestExecutable(self, filename, filepath):
        """
        Check if a file is a libc test executable we should run.

        Recognized patterns (all must end with .__build__):
          libc.test.src.<category>.<test_name>.__build__
          libc.test.src.<category>.<test_name>.__unit__[.<opts>...].__build__
          libc.test.src.<category>.<test_name>.__hermetic__[.<opts>...].__build__
          libc.test.include.<test_name>.__unit__[.<opts>...].__build__
          libc.test.include.<test_name>.__hermetic__[.<opts>...].__build__
          libc.test.integration.<category>.<test_name>.__build__
        """
        if not filename.endswith(".__build__"):
            return False
        if filename.startswith("libc.test.src."):
            pass  # Accept all src tests ending in .__build__
        elif filename.startswith("libc.test.include."):
            if ".__unit__." not in filename and ".__hermetic__." not in filename:
                return False
        elif filename.startswith("libc.test.integration."):
            pass  # Accept all integration tests ending in .__build__
        else:
            return False
        if not os.path.isfile(filepath):
            return False
        if not os.access(filepath, os.X_OK):
            return False
        return True

    def execute(self, test, litConfig):
        """
        Execute a test by running the test executable.

        Runs from the executable's directory so relative paths (like
        testdata/test.txt) work correctly.
        """

        test_path = test.getSourcePath()
        exec_dir = os.path.dirname(test_path)

        test_cmd_template = getattr(test.config, "libc_test_cmd", "")
        if test_cmd_template:
            test_cmd = test_cmd_template.replace("@BINARY@", test_path)
            cmd_args = shlex.split(test_cmd)
            if not cmd_args:
                cmd_args = [test_path]
            out, err, exit_code = lit.util.executeCommand(cmd_args, cwd=exec_dir)
        else:
            out, err, exit_code = lit.util.executeCommand([test_path], cwd=exec_dir)

        if not exit_code:
            return lit.Test.PASS, ""

        return lit.Test.FAIL, out + err

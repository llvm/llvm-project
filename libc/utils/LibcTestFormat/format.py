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

Test executables are discovered by looking for files matching:
  libc.test.src.<category>.<test_name>.__unit__.__build__
  libc.test.src.<category>.<test_name>.__hermetic__.__build__

These are created by the add_libc_test() infrastructure.
"""

import os

import lit.formats
import lit.Test
import lit.util


class LibcTest(lit.formats.ExecutableTest):
    """
    Test format for libc unit tests.

    Extends ExecutableTest to discover tests from the build directory
    rather than the source directory. Test executables are named like:
      libc.test.src.ctype.isalnum_test.__unit__.__build__
    and return 0 on success.
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

        for filename in os.listdir(source_path):
            filepath = os.path.join(source_path, filename)

            # Match our test executable pattern
            if self._isTestExecutable(filename, filepath):
                # Create a test with the executable name
                yield lit.Test.Test(testSuite, path_in_suite + (filename,), localConfig)

    def _isTestExecutable(self, filename, filepath):
        """Check if a file is a test executable we should run."""
        # Pattern: libc.test.src.*.__unit__.__build__ or .__hermetic__.__build__
        if not filename.startswith("libc.test."):
            return False
        if not (
            filename.endswith(".__unit__.__build__")
            or filename.endswith(".__hermetic__.__build__")
        ):
            return False
        # Must be executable
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

        testPath = test.getSourcePath()
        execDir = os.path.dirname(testPath)

        out, err, exitCode = lit.util.executeCommand(testPath, cwd=execDir)

        if not exitCode:
            return lit.Test.PASS, ""

        return lit.Test.FAIL, out + err

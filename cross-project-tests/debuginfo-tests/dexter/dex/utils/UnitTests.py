# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Unit test harness."""

from fnmatch import fnmatch
import os
import unittest

from io import StringIO

from dex.utils import is_native_windows, has_pywin32
from dex.utils import PreserveAutoColors, PrettyOutput
from dex.utils import Timer


class DexTestLoader(unittest.TestLoader):
    def _match_path(self, path, full_path, pattern):
        """Don't try to import platform-specific modules for the wrong platform
        during test discovery.
        """
        d = os.path.basename(os.path.dirname(full_path))
        if is_native_windows():
            if d == "posix":
                return False
            if d == "windows":
                return has_pywin32()
        else:
            if d == "windows":
                return False
            elif d == "dbgeng":
                return False
        return fnmatch(path, pattern)


def unit_tests_ok(context):
    unittest.TestCase.maxDiff = None  # remove size limit from diff output.

    with Timer("unit tests"):
        suite = DexTestLoader().discover(context.root_directory, pattern="*.py")
        stream = StringIO()
        result = unittest.TextTestRunner(verbosity=2, stream=stream).run(suite)

        ok = result.wasSuccessful()
        if not ok or context.options.unittest == "show-all":
            with PreserveAutoColors(context.o):
                context.o.auto_reds.extend([r"FAIL(ED|\:)", r"\.\.\.\s(FAIL|ERROR)$"])
                context.o.auto_greens.extend([r"^OK$", r"\.\.\.\sok$"])
                context.o.auto_blues.extend([r"^Ran \d+ test"])
                context.o.default("\n")
                for line in stream.getvalue().splitlines(True):
                    context.o.auto(line, stream=PrettyOutput.stderr)

        return ok


class TestUnitTests(unittest.TestCase):
    def test_sanity(self):
        self.assertEqual(1, 1)

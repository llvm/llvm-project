# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
import unittest
from unittest import mock

from clang_ssaf_orchestrator_make import logging_diagnostics


class Diagnostics(unittest.TestCase):
    def _stderr(self, work):
        stream = io.StringIO()
        with mock.patch("sys.stderr", stream):
            work()
        return stream.getvalue()

    def test_an_error_says_so(self):
        self.assertEqual(
            "clang-ssaf-orchestrator-make: error: it broke\n",
            self._stderr(lambda: logging_diagnostics.report_error("it broke")),
        )

    def test_details_are_indented_under_their_error(self):
        self.assertEqual(
            "clang-ssaf-orchestrator-make: error: two reasons\n"
            "  the first\n"
            "  the second\n",
            self._stderr(
                lambda: logging_diagnostics.report_error(
                    "two reasons", ["the first", "the second"]
                )
            ),
        )

    def test_a_warning_says_so(self):
        self.assertEqual(
            "clang-ssaf-orchestrator-make: warning: careful\n",
            self._stderr(lambda: logging_diagnostics.report_warning("careful")),
        )

    def test_an_error_is_not_mistakable_for_a_note(self):
        error = self._stderr(lambda: logging_diagnostics.report_error("same words"))
        note = self._stderr(lambda: logging_diagnostics.report_note(True, "same words"))
        self.assertNotEqual(error, note)

    def test_a_note_is_silent_unless_asked_for(self):
        self.assertEqual(
            "", self._stderr(lambda: logging_diagnostics.report_note(False, "quiet"))
        )
        self.assertEqual(
            "clang-ssaf-orchestrator-make: note: loud\n",
            self._stderr(lambda: logging_diagnostics.report_note(True, "loud")),
        )


class Results(unittest.TestCase):
    def _stdout(self, work):
        stream = io.StringIO()
        with mock.patch("sys.stdout", stream):
            work()
        return stream.getvalue()

    def test_the_output_directory(self):
        self.assertEqual(
            "output-dir: /out\n",
            self._stdout(lambda: logging_diagnostics.report_output_dir("/out")),
        )

    def test_the_summaries_and_the_link_unit(self):
        self.assertEqual(
            "summaries: 3\n"
            "link-unit: p/build/prg\n"
            "link-unit-summary: /out/p/build/prg.summary.json\n",
            self._stdout(
                lambda: logging_diagnostics.report_summaries(
                    3, "p/build/prg", "/out/p/build/prg.summary.json"
                )
            ),
        )

    def test_the_analysis_result(self):
        self.assertEqual(
            "analysis-result: /out/wpa.json\n",
            self._stdout(
                lambda: logging_diagnostics.report_analysis_result("/out/wpa.json")
            ),
        )

    def test_results_are_flushed_as_they_are_written(self):
        flushes = []
        stream = io.StringIO()
        stream.flush = lambda: flushes.append(stream.getvalue())
        with mock.patch("sys.stdout", stream):
            logging_diagnostics.report_summaries(1, "prg", "/out/prg.json")
        self.assertEqual(3, len(flushes))


if __name__ == "__main__":
    unittest.main()

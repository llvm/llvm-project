# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import tempfile
import unittest

from clang_ssaf_driver import source_edits
from clang_ssaf_driver.source_edits import Replacement


def _entry(file_path, offset, length, text):
    return [
        "  - FilePath:        %s" % file_path,
        "    Offset:          %d" % offset,
        "    Length:          %d" % length,
        "    ReplacementText: %s" % text,
    ]


class Reading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, "edits.yaml")

    def _read(self, lines, line_end=b"\n"):
        with open(self.path, "wb") as handle:
            handle.write(line_end.join(line.encode("utf-8") for line in lines))
            handle.write(line_end)
        return source_edits.read_source_edits(self.path)

    def _document(self, *entries):
        lines = ["---", "MainSourceFile:  ''", "Replacements:"]
        for entry in entries:
            lines.extend(entry)
        return lines + ["..."]

    def test_every_scalar_the_merge_tool_writes_is_read(self):
        written = [
            ("plain", "plain"),
            ("a\tb", "a\tb"),
            ("' lead'", " lead"),
            ("'it''s'", "it's"),
            ("''", ""),
            ("'4'", "4"),
            ("'true'", "true"),
            ('"line\\nbreak"', "line\nbreak"),
            ("'tab\tand\"quote\\'", 'tab\tand"quote\\'),
            ('"\\x01\x7f"', "\x01\x7f"),
            ('"\u00e9"', "\u00e9"),
            ('"\U0001F600"', "\U0001F600"),
            ('"\\L\\N\\_"', "\u2028\x85\xa0"),
        ]
        replacements, problem = self._read(
            self._document(
                *[
                    _entry("'src/f.cpp'", index * 5, 1, text)
                    for index, (text, _) in enumerate(written)
                ]
            )
        )
        self.assertIsNone(problem)
        self.assertEqual(
            [
                Replacement("src/f.cpp", index * 5, 1, text)
                for index, (_, text) in enumerate(written)
            ],
            replacements,
        )

    def test_every_escape_of_a_double_quoted_scalar_is_read(self):
        replacements, problem = self._read(
            self._document(
                _entry(
                    "f.cpp",
                    0,
                    0,
                    '"\\0\\a\\b\\t\\n\\v\\f\\r\\e\\"\\\\\\x41\\u00e9\\U0001F600"',
                )
            )
        )
        self.assertIsNone(problem)
        self.assertEqual(
            '\0\a\b\t\n\v\f\r\x1b"\\A\u00e9\U0001F600', replacements[0].text
        )

    def test_no_replacements_is_an_empty_list(self):
        replacements, problem = self._read(
            ["---", "MainSourceFile:  ''", "Replacements:    []", "..."]
        )
        self.assertIsNone(problem)
        self.assertEqual([], replacements)

    def test_crlf_line_ends_are_read(self):
        replacements, problem = self._read(
            self._document(_entry("f.cpp", 1, 2, "x")), line_end=b"\r\n"
        )
        self.assertIsNone(problem)
        self.assertEqual([Replacement("f.cpp", 1, 2, "x")], replacements)

    def _problem(self, lines):
        replacements, problem = self._read(lines)
        self.assertIsNone(replacements)
        return problem.message

    def test_a_truncated_file_is_malformed(self):
        self.assertEqual(
            "malformed source edits: %s:7: expected '...'" % self.path,
            self._problem(self._document(_entry("f.cpp", 1, 2, "x"))[:-1]),
        )

    def test_a_malformed_line_is_reported_with_its_number(self):
        lines = self._document(_entry("f.cpp", 1, 2, "x"))
        lines[4] = "    Offset:          one"
        self.assertEqual(
            "malformed source edits: %s:5: expected a number, got 'one'" % self.path,
            self._problem(lines),
        )

    def test_an_unknown_escape_is_malformed(self):
        self.assertIn(
            "unknown escape '\\q'",
            self._problem(self._document(_entry("f.cpp", 1, 2, '"\\q"'))),
        )

    def test_a_replacement_missing_a_field_is_malformed(self):
        self.assertIn(
            "replacement without 'ReplacementText'",
            self._problem(self._document(_entry("f.cpp", 1, 2, "x")[:3])),
        )

    def test_a_file_that_cannot_be_read_is_a_problem(self):
        _, problem = source_edits.read_source_edits(self.path)
        self.assertIn("cannot read the source edits", problem.message)


class Writing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, "edits.yaml")

    def test_what_is_written_reads_back(self):
        replacements = [
            Replacement(os.path.join(self.tmp, "f.cpp"), 0, 3, text)
            for text in (
                "",
                "bounded_ptr<int> ",
                "it's",
                'a "quote" and a \\',
                "line\nbreak\ttab\r",
                "\x01\x7f",
                "\u00e9\U0001F600",
                "\x85\u2028\u2029",
            )
        ]
        source_edits.write_source_edits(self.path, replacements)
        self.assertEqual(
            (replacements, None), source_edits.read_source_edits(self.path)
        )

    def test_no_replacements_reads_back_as_none(self):
        source_edits.write_source_edits(self.path, [])
        self.assertEqual(([], None), source_edits.read_source_edits(self.path))


if __name__ == "__main__":
    unittest.main()

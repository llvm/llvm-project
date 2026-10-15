# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import subprocess
import tempfile
import unittest

from clang_ssaf_driver import patch
from clang_ssaf_driver.source_edits import Replacement


class ApplyingEdits(unittest.TestCase):
    def _apply(self, original, edits):
        return patch.apply_edits(original, sorted(edits))

    def test_replacements_deletions_and_insertions(self):
        self.assertEqual(
            (b"bounded_ptr<int> p; q[4]/*T*/;", None),
            self._apply(
                b"int *p; q[4];",
                [(0, 5, "bounded_ptr<int> "), (12, 0, "/*T*/")],
            ),
        )
        self.assertEqual((b"int q;", None), self._apply(b"int q[4];", [(5, 3, "")]))

    def test_an_insertion_goes_before_a_replacement_at_its_offset(self):
        self.assertEqual(
            (b"(bounded_ptr<int> )p", None),
            self._apply(b"int *p", [(0, 0, "("), (0, 5, "bounded_ptr<int> )")]),
        )

    def test_an_insertion_at_the_end_of_a_replacement_follows_it(self):
        self.assertEqual((b"Yx", None), self._apply(b"ab", [(0, 2, "Y"), (2, 0, "x")]))

    def test_overlapping_edits_are_refused(self):
        _, problem = self._apply(b"abcdef", [(0, 3, "x"), (2, 2, "y")])
        self.assertEqual("edits overlap at byte 2", problem)
        _, problem = self._apply(b"abcdef", [(1, 0, "x"), (1, 0, "y")])
        self.assertEqual("edits overlap at byte 1", problem)

    def test_an_edit_past_the_end_is_refused(self):
        _, problem = self._apply(b"abc", [(2, 2, "x")])
        self.assertEqual("an edit at byte 2 runs past the end of the file", problem)

    def test_the_text_goes_in_as_utf8(self):
        self.assertEqual(
            ("éb".encode("utf-8"), None), self._apply(b"ab", [(0, 1, "é")])
        )


class Collecting(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = os.path.join(self.tmp, "project")
        os.makedirs(os.path.join(self.root, "src"))

    def _file(self, relative, content):
        path = os.path.join(self.root, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(content)
        return path

    def _collect(self, replacements, conflicts=None):
        collected, problem = patch.collect(replacements, self.root, conflicts or {})
        self.assertIsNone(problem)
        return collected

    def test_edits_are_grouped_by_file_and_sorted(self):
        self._file("src/b.cpp", b"int *b;\n")
        self._file("src/a.cpp", b"int *a;\n")
        collected = self._collect(
            [
                Replacement("src/b.cpp", 0, 5, "bounded_ptr<int> "),
                Replacement("src/a.cpp", 0, 5, "bounded_ptr<int> "),
            ]
        )
        self.assertEqual(
            ["src/a.cpp", "src/b.cpp"], [f.relative_path for f in collected.files]
        )
        self.assertEqual(b"bounded_ptr<int> a;\n", collected.files[0].edited)
        self.assertEqual(([], []), (collected.outside, collected.blocked))

    def test_every_spelling_of_a_file_is_one_file_and_an_edit_counts_once(self):
        header = self._file("src/h.h", b"int *g;\n")
        os.link(header, os.path.join(self.root, "src", "linked.h"))
        edit = (0, 5, "bounded_ptr<int> ")
        collected = self._collect(
            [
                Replacement("src/h.h", *edit),
                Replacement("src/../src/h.h", *edit),
                Replacement(header, *edit),
                Replacement("src/linked.h", *edit),
            ]
        )
        self.assertEqual(["src/h.h"], [f.relative_path for f in collected.files])
        self.assertEqual([edit], collected.files[0].edits)

    def test_an_edit_that_overlaps_a_reported_conflict_is_left_out(self):
        header = self._file("src/h.h", b"0123456789abcdef")
        collected = self._collect(
            [
                Replacement("src/h.h", 0, 2, "A"),
                Replacement("src/h.h", 2, 0, "B"),
                Replacement("src/h.h", 4, 4, "C"),
                Replacement("src/h.h", 7, 0, "D"),
                Replacement("src/h.h", 10, 2, "E"),
            ],
            {header: [(2, 9)]},
        )
        self.assertEqual(
            [(0, 2, "A"), (2, 0, "B"), (10, 2, "E")], collected.files[0].edits
        )
        self.assertEqual([("src/h.h", 2)], collected.blocked)

    def test_a_file_whose_edits_all_conflict_is_left_out(self):
        header = self._file("src/h.h", b"int *g;\n")
        collected = self._collect(
            [Replacement("src/h.h", 0, 5, "bounded_ptr<int> ")], {header: [(0, 5)]}
        )
        self.assertEqual([], collected.files)
        self.assertEqual([("src/h.h", 1)], collected.blocked)

    def test_a_file_outside_the_project_is_left_out(self):
        self._file("src/a.cpp", b"int *a;\n")
        elsewhere = os.path.join(self.tmp, "elsewhere.h")
        with open(elsewhere, "wb") as handle:
            handle.write(b"int *x;\n")
        collected = self._collect(
            [
                Replacement("src/a.cpp", 0, 5, "bounded_ptr<int> "),
                Replacement(elsewhere, 0, 5, "bounded_ptr<int> "),
            ]
        )
        self.assertEqual(["src/a.cpp"], [f.relative_path for f in collected.files])
        self.assertEqual([elsewhere], collected.outside)

    def test_every_file_that_cannot_be_edited_is_reported_at_once(self):
        self._file("src/a.cpp", b"abc")
        collected, problem = patch.collect(
            [
                Replacement("src/a.cpp", 2, 5, "x"),
                Replacement("src/missing.cpp", 0, 0, "x"),
            ],
            self.root,
            {},
        )
        self.assertIsNone(collected)
        self.assertEqual("the source edits cannot be applied:", problem.message)
        self.assertEqual(
            [
                "src/a.cpp: an edit at byte 2 runs past the end of the file",
                "src/missing.cpp: No such file or directory",
            ],
            list(problem.details),
        )


class Rendering(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _render(self, original, edits, name="src/a.cpp"):
        edited, problem = patch.apply_edits(original, edits)
        self.assertIsNone(problem)
        path = os.path.join(self.tmp, name)
        return patch.render([patch.FileEdits(name, path, edits, original, edited)])

    def test_a_unified_diff_with_paths_under_a_and_b(self):
        self.assertEqual(
            b"--- a/src/a.cpp\n"
            b"+++ b/src/a.cpp\n"
            b"@@ -1,3 +1,3 @@\n"
            b" int f();\n"
            b"-int *p;\n"
            b"+bounded_ptr<int> p;\n"
            b" int g();\n",
            self._render(
                b"int f();\nint *p;\nint g();\n", [(9, 5, "bounded_ptr<int> ")]
            ),
        )

    def test_a_missing_final_newline_is_marked(self):
        self.assertEqual(
            b"--- a/src/a.cpp\n"
            b"+++ b/src/a.cpp\n"
            b"@@ -1 +1 @@\n"
            b"-int *p;\n"
            b"\\ No newline at end of file\n"
            b"+bounded_ptr<int> p;\n"
            b"\\ No newline at end of file\n",
            self._render(b"int *p;", [(0, 5, "bounded_ptr<int> ")]),
        )

    def test_line_ends_and_bytes_are_kept_as_they_are(self):
        rendered = self._render(
            b"\xff\r\nint *p;\r\nold\rmac\n", [(3, 5, "bounded_ptr<int> ")]
        )
        self.assertIn(b"-int *p;\r\n+bounded_ptr<int> p;\r\n", rendered)
        self.assertIn(b" old\rmac\n", rendered)

    def test_no_edits_is_an_empty_patch(self):
        self.assertEqual(b"", patch.render([]))

    @unittest.skipUnless(shutil.which("patch"), "needs patch")
    def test_patch_applies_what_was_rendered(self):
        cases = {
            "src/a.cpp": (b"int f();\nint *p;\nint g();\n", [(9, 5, "q ")]),
            "src/b.cpp": (b"int *p;", [(0, 5, "bounded_ptr<int> ")]),
            "src/c.cpp": (b"a\r\nint *p;\r\nb\rc\n", [(3, 5, "x")]),
        }
        files = []
        for name, (original, edits) in sorted(cases.items()):
            path = os.path.join(self.tmp, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(original)
            edited, _ = patch.apply_edits(original, edits)
            files.append(patch.FileEdits(name, path, edits, original, edited))
        subprocess.run(
            ["patch", "-p1", "-s"],
            input=patch.render(files),
            cwd=self.tmp,
            check=True,
        )
        for each in files:
            with open(each.path, "rb") as handle:
                self.assertEqual(each.edited, handle.read())


if __name__ == "__main__":
    unittest.main()

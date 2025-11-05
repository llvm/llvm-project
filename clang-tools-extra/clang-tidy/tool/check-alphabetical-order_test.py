# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# To run these tests:
# python3 check-alphabetical-order_test.py -v

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr
import importlib.util
import importlib.machinery
from typing import Any, cast


def _load_script_module():
    here = os.path.dirname(cast(str, __file__))
    script_path = os.path.normpath(os.path.join(here, "check-alphabetical-order.py"))
    loader = importlib.machinery.SourceFileLoader(
        "check_alphabetical_order", cast(str, script_path)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = cast(Any, _load_script_module())


class TestAlphabeticalOrderCheck(unittest.TestCase):
    def test_normalize_list_rst_sorts_rows(self):
        lines = [
            "Header\n",
            "\n",
            ".. csv-table:: Clang-Tidy checks\n",
            '   :header: "Check", "Info"\n',
            "\n",
            "   :doc:`zebra <clang-tidy/checks/zebra>` Z line\n",
            "   :doc:`Alpha <clang-tidy/checks/alpha>` A line\n",
            "   some non-doc row that should stay after docs\n",
            "\n",
            "Footer\n",
        ]

        out = _mod.normalize_list_rst(lines)
        # Alpha should appear before zebra in normalized csv-table rows.
        alpha: str = "Alpha <clang-tidy/checks/alpha>"
        zebra: str = "zebra <clang-tidy/checks/zebra>"
        self.assertLess(out.find(alpha), out.find(zebra))
        # Non-doc row should remain after doc rows within the table region.
        self.assertGreater(out.find("some non-doc row"), out.find("zebra"))

    def test_find_heading(self):
        lines = [
            "- something\n",
            "New checks\n",
            "^^^^^^^^^^^\n",
            "- something\n",
        ]
        idx = _mod.find_heading(lines, "New checks")
        self.assertEqual(idx, 1)

    def test_collect_and_sort_blocks(self):
        # Section content with two bullets and a suffix line.
        lines = [
            "Intro\n",
            "- :doc:`Zed <clang-tidy/checks/zed>`: details\n",
            "  continuation\n",
            "- :doc:`alpha <clang-tidy/checks/alpha>`: more details\n",
            "\n",
        ]
        prefix, blocks, suffix = _mod.collect_bullet_blocks(lines, 0, len(lines))
        # Prefix is the intro line until first bullet; suffix is trailing lines.
        self.assertEqual(prefix, ["Intro\n"])
        self.assertEqual(suffix, [])
        sorted_blocks = _mod.sort_blocks(blocks)
        joined = "".join([l for b in sorted_blocks for l in b])
        # Uppercase Z sorts before lowercase a in ASCII.
        zed: str = "Zed <clang-tidy/checks/zed>"
        aval: str = "alpha <clang-tidy/checks/alpha>"
        self.assertLess(joined.find(zed), joined.find(aval))

    def test_normalize_single_section_orders_bullets(self):
        content = [
            "New checks\n",
            "^^^^^^^^^^^\n",
            "- :doc:`zed <clang-tidy/checks/zed>`: new\n",
            "- :doc:`Alpha <clang-tidy/checks/alpha>`: new\n",
            "\n",
        ]
        out = "".join(
            _mod._normalize_release_notes_section(content, "New checks", None)
        )
        # Uppercase A sorts before lowercase z in ASCII.
        aitem: str = "- :doc:`Alpha <clang-tidy/checks/alpha>`: new"
        zitem: str = "- :doc:`zed <clang-tidy/checks/zed>`: new"
        self.assertLess(out.find(aitem), out.find(zitem))

    def test_duplicate_detection_and_report(self):
        lines = [
            "Changes in existing checks\n",
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
            "- :doc:`Alpha <clang-tidy/checks/alpha>`: change one\n",
            "- :doc:`Zed <clang-tidy/checks/zed>`: change something\n",
            "- :doc:`Alpha <clang-tidy/checks/alpha>`: change two\n",
            "\n",
        ]
        dups = _mod.find_duplicate_block_details(lines, "Changes in existing checks")
        # Expect one duplicate group for 'Alpha' with two occurrences.
        self.assertEqual(len(dups), 1)
        key, occs = dups[0]
        self.assertEqual(key.strip(), "Alpha")
        self.assertEqual(len(occs), 2)

        report = _mod._emit_duplicate_report(lines, "Changes in existing checks")
        self.assertIsInstance(report, str)
        self.assertIn("Duplicate entries in 'Changes in existing checks':", report)
        self.assertIn("-- Duplicate: Alpha", report)
        self.assertEqual(report.count("- At line "), 2)

    def test_handle_release_notes_out_unsorted_returns_ok(self):
        # When content is not normalized, the function writes normalized text and returns 0.
        rn_lines = [
            "New checks\n",
            "^^^^^^^^^^^\n",
            "- :doc:`Zed <clang-tidy/checks/zed>`: new\n",
            "- :doc:`Alpha <clang-tidy/checks/alpha>`: new\n",
            "\n",
        ]
        with tempfile.TemporaryDirectory() as td:
            rn_doc = os.path.join(td, "ReleaseNotes.rst")
            out_path = os.path.join(td, "out.rst")
            with open(rn_doc, "w", encoding="utf-8") as f:
                f.write("".join(rn_lines))

            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = _mod._handle_release_notes_out(out_path, rn_doc)

            self.assertEqual(rc, 0)
            with open(out_path, "r", encoding="utf-8") as f:
                out = f.read()

            self.assertLess(
                out.find("Alpha <clang-tidy/checks/alpha>"),
                out.find("Zed <clang-tidy/checks/zed>"),
            )
            self.assertIn("not normalized", buf.getvalue())

    def test_handle_release_notes_out_duplicates_fail(self):
        # Sorting is already correct but duplicates exist, should return 3 and report.
        rn_lines = [
            "Changes in existing checks\n",
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
            "- :doc:`Alpha <clang-tidy/checks/alpha>`: change one\n",
            "  change one\n\n",
            "- :doc:`Alpha <clang-tidy/checks/alpha>`: change two\n\n",
            "  change two\n\n",
        ]
        with tempfile.TemporaryDirectory() as td:
            rn_doc = os.path.join(td, "ReleaseNotes.rst")
            out_path = os.path.join(td, "out.rst")
            with open(rn_doc, "w", encoding="utf-8") as f:
                f.write("".join(rn_lines))

            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = _mod._handle_release_notes_out(out_path, rn_doc)

            self.assertEqual(rc, 3)
            self.assertIn(
                "Duplicate entries in 'Changes in existing checks':",
                buf.getvalue(),
            )

    def test_handle_checks_list_out_writes_normalized(self):
        list_lines = [
            ".. csv-table:: List\n",
            '   :header: "Check", "Info"\n',
            "\n",
            "   :doc:`Zed <clang-tidy/checks/zed>` foo\n",
            "   :doc:`Beta <clang-tidy/checks/beta>` bar\n",
            "   :doc:`Alpha <clang-tidy/checks/alpha>` baz\n",
            "   :doc:`Baz <clang-tidy/checks/baz>` baz\n",
        ]
        with tempfile.TemporaryDirectory() as td:
            in_doc = os.path.join(td, "list.rst")
            out_doc = os.path.join(td, "out.rst")
            with open(in_doc, "w", encoding="utf-8") as f:
                f.write("".join(list_lines))
            rc = _mod._handle_checks_list_out(out_doc, in_doc)
            self.assertEqual(rc, 0)
            with open(out_doc, "r", encoding="utf-8") as f:
                out = f.read()
            alpha = out.find("Alpha <clang-tidy/checks/alpha>")
            baz = out.find("Baz <clang-tidy/checks/baz>")
            beta = out.find("Beta <clang-tidy/checks/beta>")
            zed = out.find("Zed <clang-tidy/checks/zed>")
            for pos in (alpha, baz, beta, zed):
                self.assertGreaterEqual(pos, 0)
            self.assertLess(alpha, baz)
            self.assertLess(baz, beta)
            self.assertLess(beta, zed)


if __name__ == "__main__":
    unittest.main()

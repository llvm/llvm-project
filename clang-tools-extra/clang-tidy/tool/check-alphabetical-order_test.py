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
            "------" "\n",
            ".. csv-table:: Clang-Tidy checks\n",
            '   :header: "Name", "Offers fixes"\n',
            "\n",
            '   :doc:`bugprone-virtual-near-miss <bugprone/virtual-near-miss>`, "Yes"\n',
            "   :doc:`cert-flp30-c <cert/flp30-c>`,\n",
            '   :doc:`abseil-cleanup-ctad <abseil/cleanup-ctad>`, "Yes"\n',
            "   A non-doc row that should stay after docs\n",
            "\n",
            "Footer\n",
        ]

        out = _mod.normalize_list_rst(lines)
        pos_abseil = out.find("abseil-cleanup-ctad")
        pos_bugprone = out.find("bugprone-virtual-near-miss")
        pos_cert = out.find("cert-flp30-c")
        self.assertTrue(all(p != -1 for p in [pos_abseil, pos_bugprone, pos_cert]))
        self.assertLess(pos_abseil, pos_bugprone)
        self.assertLess(pos_bugprone, pos_cert)
        # Non-doc row should remain after doc rows within the table region.
        self.assertGreater(out.find("A non-doc row"), out.find("cert-flp30-c"))

    def test_find_heading(self):
        lines = [
            "- Deprecated the :program:`clang-tidy` ``zircon`` module. All checks have been\n",
            "  moved to the ``fuchsia`` module instead. The ``zircon`` module will be removed\n",
            "  in the 24th release.\n",
            "\n",
            "New checks\n",
            "^^^^^^^^^^\n",
            "- New :doc:`bugprone-derived-method-shadowing-base-method\n",
            "  <clang-tidy/checks/bugprone/derived-method-shadowing-base-method>` check.\n",
        ]
        idx = _mod.find_heading(lines, "New checks")
        self.assertEqual(idx, 4)

    def test_duplicate_detection_and_report(self):
        # Ensure duplicate detection works properly when sorting is incorrect.
        lines = [
            "Changes in existing checks\n",
            "^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
            "\n",
            "- Improved :doc:`bugprone-easily-swappable-parameters\n",
            "  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by\n",
            "  correcting a spelling mistake on its option\n",
            "  ``NamePrefixSuffixSilenceDissimilarityTreshold``.\n",
            "\n",
            "- Improved :doc:`bugprone-exception-escape\n",
            "  <clang-tidy/checks/bugprone/exception-escape>` check's handling of lambdas:\n",
            "  exceptions from captures are now diagnosed, exceptions in the bodies of\n",
            "  lambdas that aren't actually invoked are not.\n",
            "\n",
            "- Improved :doc:`bugprone-easily-swappable-parameters\n",
            "  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by\n",
            "  correcting a spelling mistake on its option\n",
            "  ``NamePrefixSuffixSilenceDissimilarityTreshold``.\n",
            "\n",
        ]
        dups = _mod.find_duplicate_entries(lines, "Changes in existing checks")
        # Expect one duplicate group for 'bugprone-easily-swappable-parameters' with two occurrences.
        self.assertEqual(len(dups), 1)
        key, occs = dups[0]
        self.assertEqual(
            key.strip(), "- Improved :doc:`bugprone-easily-swappable-parameters"
        )
        self.assertEqual(len(occs), 2)

        report = _mod._emit_duplicate_report(lines, "Changes in existing checks")
        self.assertIsInstance(report, str)
        self.assertIn("Duplicate entries in 'Changes in existing checks':", report)
        self.assertIn(
            "-- Duplicate: - Improved :doc:`bugprone-easily-swappable-parameters",
            report,
        )
        self.assertEqual(report.count("- At line "), 2)
        self.assertIn("- At line 4:", report)
        self.assertIn("- At line 14:", report)

    def test_process_release_notes_with_unsorted_content(self):
        # When content is not normalized, the function writes normalized text and returns 0.
        rn_lines = [
            "New checks\n",
            "^^^^^^^^^^\n",
            "\n",
            "- New :doc:`readability-redundant-parentheses\n",
            "  <clang-tidy/checks/readability/redundant-parentheses>` check.\n",
            "\n",
            "  Detect redundant parentheses.\n",
            "\n",
            "- New :doc:`bugprone-derived-method-shadowing-base-method\n",
            "  <clang-tidy/checks/bugprone/derived-method-shadowing-base-method>` check.\n",
            "\n",
            "  Finds derived class methods that shadow a (non-virtual) base class method.\n",
            "\n",
        ]
        with tempfile.TemporaryDirectory() as td:
            rn_doc = os.path.join(td, "ReleaseNotes.rst")
            out_path = os.path.join(td, "out.rst")
            with open(rn_doc, "w", encoding="utf-8") as f:
                f.write("".join(rn_lines))

            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = _mod.process_release_notes(out_path, rn_doc)

            self.assertEqual(rc, 0)
            with open(out_path, "r", encoding="utf-8") as f:
                out = f.read()

            bugprone_item = [
                "- New :doc:`bugprone-derived-method-shadowing-base-method",
                "  <clang-tidy/checks/bugprone/derived-method-shadowing-base-method>` check.",
                "  Finds derived class methods that shadow a (non-virtual) base class method.",
            ]
            readability_item = [
                "- New :doc:`readability-redundant-parentheses",
                "  <clang-tidy/checks/readability/redundant-parentheses>` check.",
                "  Detect redundant parentheses.",
            ]

            p_bugprone = [out.find(s) for s in bugprone_item]
            p_readability = [out.find(s) for s in readability_item]

            self.assertTrue(all(p != -1 for p in p_bugprone))
            self.assertTrue(all(p != -1 for p in p_readability))

            self.assertLess(p_bugprone[0], p_bugprone[1])
            self.assertLess(p_bugprone[1], p_bugprone[2])

            self.assertLess(p_readability[0], p_readability[1])
            self.assertLess(p_readability[1], p_readability[2])

            self.assertLess(
                out.find("bugprone-derived-method-shadowing-base-method"),
                out.find("readability-redundant-parentheses"),
            )
            self.assertIn("not normalized", buf.getvalue())

    def test_process_release_notes_prioritizes_sorting_over_duplicates(self):
        # Sorting is incorrect and duplicates exist, should report ordering issues first.
        rn_lines = [
            "Changes in existing checks\n",
            "^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
            "\n",
            "- Improved :doc:`bugprone-easily-swappable-parameters\n",
            "  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by\n",
            "  correcting a spelling mistake on its option\n",
            "  ``NamePrefixSuffixSilenceDissimilarityTreshold``.\n",
            "\n",
            "- Improved :doc:`bugprone-exception-escape\n",
            "  <clang-tidy/checks/bugprone/exception-escape>` check's handling of lambdas:\n",
            "  exceptions from captures are now diagnosed, exceptions in the bodies of\n",
            "  lambdas that aren't actually invoked are not.\n",
            "\n",
            "- Improved :doc:`bugprone-easily-swappable-parameters\n",
            "  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by\n",
            "  correcting a spelling mistake on its option\n",
            "  ``NamePrefixSuffixSilenceDissimilarityTreshold``.\n",
            "\n",
        ]
        with tempfile.TemporaryDirectory() as td:
            rn_doc = os.path.join(td, "ReleaseNotes.rst")
            out_path = os.path.join(td, "out.rst")
            with open(rn_doc, "w", encoding="utf-8") as f:
                f.write("".join(rn_lines))

            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = _mod.process_release_notes(out_path, rn_doc)
            self.assertEqual(rc, 0)
            self.assertIn(
                "Note: 'ReleaseNotes.rst' is not normalized; Please fix ordering first.",
                buf.getvalue(),
            )

    def test_process_release_notes_with_duplicates_fails(self):
        # Sorting is already correct but duplicates exist, should return 3 and report.
        rn_lines = [
            "Changes in existing checks\n",
            "^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
            "\n",
            "- Improved :doc:`bugprone-easily-swappable-parameters\n",
            "  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by\n",
            "  correcting a spelling mistake on its option\n",
            "  ``NamePrefixSuffixSilenceDissimilarityTreshold``.\n",
            "\n",
            "- Improved :doc:`bugprone-easily-swappable-parameters\n",
            "  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by\n",
            "  correcting a spelling mistake on its option\n",
            "  ``NamePrefixSuffixSilenceDissimilarityTreshold``.\n",
            "\n",
            "- Improved :doc:`bugprone-exception-escape\n",
            "  <clang-tidy/checks/bugprone/exception-escape>` check's handling of lambdas:\n",
            "  exceptions from captures are now diagnosed, exceptions in the bodies of\n",
            "  lambdas that aren't actually invoked are not.\n",
            "\n",
        ]
        with tempfile.TemporaryDirectory() as td:
            rn_doc = os.path.join(td, "ReleaseNotes.rst")
            out_path = os.path.join(td, "out.rst")
            with open(rn_doc, "w", encoding="utf-8") as f:
                f.write("".join(rn_lines))

            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = _mod.process_release_notes(out_path, rn_doc)

            self.assertEqual(rc, 3)
            self.assertIn(
                "-- Duplicate: - Improved :doc:`bugprone-easily-swappable-parameters",
                buf.getvalue(),
            )

    def test_process_checks_list_normalizes_output(self):
        list_lines = [
            ".. csv-table:: List\n",
            '   :header: "Name", "Redirect", "Offers fixes"\n',
            "\n",
            '   :doc:`cert-dcl16-c <cert/dcl16-c>`, :doc:`readability-uppercase-literal-suffix <readability/uppercase-literal-suffix>`, "Yes"\n',
            "   :doc:`cert-con36-c <cert/con36-c>`, :doc:`bugprone-spuriously-wake-up-functions <bugprone/spuriously-wake-up-functions>`,\n",
            '   :doc:`cert-dcl37-c <cert/dcl37-c>`, :doc:`bugprone-reserved-identifier <bugprone/reserved-identifier>`, "Yes"\n',
            "   :doc:`cert-arr39-c <cert/arr39-c>`, :doc:`bugprone-sizeof-expression <bugprone/sizeof-expression>`,\n",
        ]
        with tempfile.TemporaryDirectory() as td:
            in_doc = os.path.join(td, "list.rst")
            out_doc = os.path.join(td, "out.rst")
            with open(in_doc, "w", encoding="utf-8") as f:
                f.write("".join(list_lines))
            rc = _mod.process_checks_list(out_doc, in_doc)
            self.assertEqual(rc, 0)
            with open(out_doc, "r", encoding="utf-8") as f:
                out = f.read()
            dcl16_pos = out.find("cert-dcl16-c <cert/dcl16-c>")
            con36_pos = out.find("cert-con36-c <cert/con36-c>")
            dcl37_pos = out.find("cert-dcl37-c <cert/dcl37-c>")
            arr39_pos = out.find("cert-arr39-c <cert/arr39-c>")
            for pos in (dcl16_pos, con36_pos, dcl37_pos, arr39_pos):
                self.assertGreaterEqual(pos, 0)
            self.assertLess(arr39_pos, con36_pos)
            self.assertLess(con36_pos, dcl16_pos)
            self.assertLess(dcl16_pos, dcl37_pos)


if __name__ == "__main__":
    unittest.main()

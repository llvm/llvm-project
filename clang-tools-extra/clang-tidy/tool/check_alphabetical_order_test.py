# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# To run these tests:
# python3 check_alphabetical_order_test.py -v

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from typing import cast


import check_alphabetical_order as _mod


class TestAlphabeticalOrderCheck(unittest.TestCase):
    def test_normalize_list_rst_sorts_rows(self):
        input_lines = [
            ".. csv-table:: Clang-Tidy checks\n",
            '   :header: "Name", "Offers fixes"\n',
            "\n",
            '   :doc:`bugprone-virtual-near-miss <bugprone/virtual-near-miss>`, "Yes"\n',
            "   :doc:`cert-flp30-c <cert/flp30-c>`,\n",
            '   :doc:`abseil-cleanup-ctad <abseil/cleanup-ctad>`, "Yes"\n',
            "   A non-doc row that should stay after docs\n",
        ]

        expected_lines = [
            ".. csv-table:: Clang-Tidy checks\n",
            '   :header: "Name", "Offers fixes"\n',
            "\n",
            '   :doc:`abseil-cleanup-ctad <abseil/cleanup-ctad>`, "Yes"\n',
            '   :doc:`bugprone-virtual-near-miss <bugprone/virtual-near-miss>`, "Yes"\n',
            "   :doc:`cert-flp30-c <cert/flp30-c>`,\n",
            "   A non-doc row that should stay after docs\n",
        ]

        out_str = _mod.normalize_list_rst("".join(input_lines))
        self.assertEqual(out_str, "".join(expected_lines))

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
        report = _mod._emit_duplicate_report(lines, "Changes in existing checks")
        self.assertIsNotNone(report)
        report_str = cast(str, report)

        expected_report = "".join(
            [
                "Error: Duplicate entries in 'Changes in existing checks':\n",
                "\n",
                "-- Duplicate: - Improved :doc:`bugprone-easily-swappable-parameters\n",
                "\n",
                "- At line 4:\n",
                "- Improved :doc:`bugprone-easily-swappable-parameters\n",
                "  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by\n",
                "  correcting a spelling mistake on its option\n",
                "  ``NamePrefixSuffixSilenceDissimilarityTreshold``.\n",
                "\n",
                "- At line 14:\n",
                "- Improved :doc:`bugprone-easily-swappable-parameters\n",
                "  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by\n",
                "  correcting a spelling mistake on its option\n",
                "  ``NamePrefixSuffixSilenceDissimilarityTreshold``.\n",
            ]
        )
        self.assertEqual(report_str, expected_report)

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

            expected_out = "".join(
                [
                    "New checks\n",
                    "^^^^^^^^^^\n",
                    "\n",
                    "- New :doc:`bugprone-derived-method-shadowing-base-method\n",
                    "  <clang-tidy/checks/bugprone/derived-method-shadowing-base-method>` check.\n",
                    "\n",
                    "  Finds derived class methods that shadow a (non-virtual) base class method.\n",
                    "\n",
                    "- New :doc:`readability-redundant-parentheses\n",
                    "  <clang-tidy/checks/readability/redundant-parentheses>` check.\n",
                    "\n",
                    "  Detect redundant parentheses.\n",
                    "\n",
                    "\n",
                ]
            )

            self.assertEqual(out, expected_out)
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

            with open(out_path, "r", encoding="utf-8") as f:
                out = f.read()
            expected_out = "".join(
                [
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
                    "\n",
                ]
            )
            self.assertEqual(out, expected_out)

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
            expected_report = "".join(
                [
                    "Error: Duplicate entries in 'Changes in existing checks':\n",
                    "\n",
                    "-- Duplicate: - Improved :doc:`bugprone-easily-swappable-parameters\n",
                    "\n",
                    "- At line 4:\n",
                    "- Improved :doc:`bugprone-easily-swappable-parameters\n",
                    "  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by\n",
                    "  correcting a spelling mistake on its option\n",
                    "  ``NamePrefixSuffixSilenceDissimilarityTreshold``.\n",
                    "\n",
                    "- At line 9:\n",
                    "- Improved :doc:`bugprone-easily-swappable-parameters\n",
                    "  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by\n",
                    "  correcting a spelling mistake on its option\n",
                    "  ``NamePrefixSuffixSilenceDissimilarityTreshold``.\n",
                    "\n",
                ]
            )
            self.assertEqual(buf.getvalue(), expected_report)

            with open(out_path, "r", encoding="utf-8") as f:
                out = f.read()
            self.assertEqual(out, "".join(rn_lines))

    def test_release_notes_handles_nested_sub_bullets(self):
        rn_lines = [
            "Changes in existing checks\n",
            "^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
            "\n",
            "- Improved :doc:`bugprone-easily-swappable-parameters\n",
            "  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by\n",
            "  correcting a spelling mistake on its option\n",
            "  ``NamePrefixSuffixSilenceDissimilarityTreshold``.\n",
            "\n",
            "- Improved :doc:`llvm-prefer-isa-or-dyn-cast-in-conditionals\n",
            "  <clang-tidy/checks/llvm/prefer-isa-or-dyn-cast-in-conditionals>` check:\n",
            "\n",
            "  - Fix-it handles callees with nested-name-specifier correctly.\n",
            "\n",
            "  - ``if`` statements with init-statement (``if (auto X = ...; ...)``) are\n",
            "    handled correctly.\n",
            "\n",
            "  - ``for`` loops are supported.\n",
            "\n",
            "- Improved :doc:`bugprone-exception-escape\n",
            "  <clang-tidy/checks/bugprone/exception-escape>` check's handling of lambdas:\n",
            "  exceptions from captures are now diagnosed, exceptions in the bodies of\n",
            "  lambdas that aren't actually invoked are not.\n",
            "\n",
        ]

        out = _mod.normalize_release_notes(rn_lines)

        expected_out = "".join(
            [
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
                "- Improved :doc:`llvm-prefer-isa-or-dyn-cast-in-conditionals\n",
                "  <clang-tidy/checks/llvm/prefer-isa-or-dyn-cast-in-conditionals>` check:\n",
                "\n",
                "  - Fix-it handles callees with nested-name-specifier correctly.\n",
                "\n",
                "  - ``if`` statements with init-statement (``if (auto X = ...; ...)``) are\n",
                "    handled correctly.\n",
                "\n",
                "  - ``for`` loops are supported.\n",
                "\n\n",
            ]
        )
        self.assertEqual(out, expected_out)

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

            expected_out = "".join(
                [
                    ".. csv-table:: List\n",
                    '   :header: "Name", "Redirect", "Offers fixes"\n',
                    "\n",
                    "   :doc:`cert-arr39-c <cert/arr39-c>`, :doc:`bugprone-sizeof-expression <bugprone/sizeof-expression>`,\n",
                    "   :doc:`cert-con36-c <cert/con36-c>`, :doc:`bugprone-spuriously-wake-up-functions <bugprone/spuriously-wake-up-functions>`,\n",
                    '   :doc:`cert-dcl16-c <cert/dcl16-c>`, :doc:`readability-uppercase-literal-suffix <readability/uppercase-literal-suffix>`, "Yes"\n',
                    '   :doc:`cert-dcl37-c <cert/dcl37-c>`, :doc:`bugprone-reserved-identifier <bugprone/reserved-identifier>`, "Yes"\n',
                ]
            )
            self.assertEqual(out, expected_out)


if __name__ == "__main__":
    unittest.main()

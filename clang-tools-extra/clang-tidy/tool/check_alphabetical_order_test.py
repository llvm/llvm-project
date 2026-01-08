# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# To run these tests:
# python3 check_alphabetical_order_test.py -v

import check_alphabetical_order as _mod
from contextlib import redirect_stderr
import io
import os
import tempfile
import textwrap
from typing import cast
import unittest


class TestAlphabeticalOrderCheck(unittest.TestCase):
    def test_normalize_list_rst_sorts_rows(self) -> None:
        input_text = textwrap.dedent(
            """\
            .. csv-table:: Clang-Tidy checks
               :header: "Name", "Offers fixes"

               :doc:`bugprone-virtual-near-miss <bugprone/virtual-near-miss>`, "Yes"
               :doc:`cert-flp30-c <cert/flp30-c>`,
               :doc:`abseil-cleanup-ctad <abseil/cleanup-ctad>`, "Yes"
               A non-doc row that should stay after docs
            """
        )

        expected_text = textwrap.dedent(
            """\
            .. csv-table:: Clang-Tidy checks
               :header: "Name", "Offers fixes"

               :doc:`abseil-cleanup-ctad <abseil/cleanup-ctad>`, "Yes"
               :doc:`bugprone-virtual-near-miss <bugprone/virtual-near-miss>`, "Yes"
               :doc:`cert-flp30-c <cert/flp30-c>`,
               A non-doc row that should stay after docs
            """
        )

        out_str = _mod.normalize_list_rst(input_text)
        self.assertEqual(out_str, expected_text)

    def test_find_heading(self) -> None:
        text = textwrap.dedent(
            """\
            - Deprecated the :program:`clang-tidy` ``zircon`` module. All checks have been
              moved to the ``fuchsia`` module instead. The ``zircon`` module will be removed
              in the 24th release.

            New checks
            ^^^^^^^^^^
            - New :doc:`bugprone-derived-method-shadowing-base-method
              <clang-tidy/checks/bugprone/derived-method-shadowing-base-method>` check.
            """
        )
        lines = text.splitlines(True)
        idx = _mod.find_heading(lines, "New checks")
        self.assertEqual(idx, 4)

    def test_duplicate_detection_and_report(self) -> None:
        # Ensure duplicate detection works properly when sorting is incorrect.
        text = textwrap.dedent(
            """\
            Changes in existing checks
            ^^^^^^^^^^^^^^^^^^^^^^^^^^

            - Improved :doc:`bugprone-easily-swappable-parameters
              <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by
              correcting a spelling mistake on its option
              ``NamePrefixSuffixSilenceDissimilarityTreshold``.

            - Improved :doc:`bugprone-exception-escape
              <clang-tidy/checks/bugprone/exception-escape>` check's handling of lambdas:
              exceptions from captures are now diagnosed, exceptions in the bodies of
              lambdas that aren't actually invoked are not.

            - Improved :doc:`bugprone-easily-swappable-parameters
              <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by
              correcting a spelling mistake on its option
              ``NamePrefixSuffixSilenceDissimilarityTreshold``.

            """
        )
        lines = text.splitlines(True)
        report = _mod._emit_duplicate_report(lines, "Changes in existing checks")
        self.assertIsNotNone(report)
        report_str = cast(str, report)

        expected_report = textwrap.dedent(
            """\
            Error: Duplicate entries in 'Changes in existing checks'.

            Please merge these entries into a single bullet point.

            -- Duplicate: - Improved :doc:`bugprone-easily-swappable-parameters

            - At line 4:
            - Improved :doc:`bugprone-easily-swappable-parameters
              <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by
              correcting a spelling mistake on its option
              ``NamePrefixSuffixSilenceDissimilarityTreshold``.

            - At line 14:
            - Improved :doc:`bugprone-easily-swappable-parameters
              <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by
              correcting a spelling mistake on its option
              ``NamePrefixSuffixSilenceDissimilarityTreshold``.
            """
        )
        self.assertEqual(report_str, expected_report)

    def test_process_release_notes_with_unsorted_content(self) -> None:
        # When content is not normalized, the function writes normalized text and returns 0.
        rn_text = textwrap.dedent(
            """\
            New checks
            ^^^^^^^^^^

            - New :doc:`readability-redundant-parentheses
              <clang-tidy/checks/readability/redundant-parentheses>` check.

              Detect redundant parentheses.

            - New :doc:`bugprone-derived-method-shadowing-base-method
              <clang-tidy/checks/bugprone/derived-method-shadowing-base-method>` check.

              Finds derived class methods that shadow a (non-virtual) base class method.

            """
        )
        with tempfile.TemporaryDirectory() as td:
            rn_doc = os.path.join(td, "ReleaseNotes.rst")
            out_path = os.path.join(td, "out.rst")
            with open(rn_doc, "w", encoding="utf-8") as f:
                f.write(rn_text)

            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = _mod.process_release_notes(out_path, rn_doc)

            self.assertEqual(rc, 0)
            with open(out_path, "r", encoding="utf-8") as f:
                out = f.read()

            expected_out = textwrap.dedent(
                """\
                New checks
                ^^^^^^^^^^

                - New :doc:`bugprone-derived-method-shadowing-base-method
                  <clang-tidy/checks/bugprone/derived-method-shadowing-base-method>` check.

                  Finds derived class methods that shadow a (non-virtual) base class method.

                - New :doc:`readability-redundant-parentheses
                  <clang-tidy/checks/readability/redundant-parentheses>` check.

                  Detect redundant parentheses.


                """
            )

            self.assertEqual(out, expected_out)
            self.assertIn("not alphabetically sorted", buf.getvalue())

    def test_process_release_notes_prioritizes_sorting_over_duplicates(self) -> None:
        # Sorting is incorrect and duplicates exist, should report ordering issues first.
        rn_text = textwrap.dedent(
            """\
            Changes in existing checks
            ^^^^^^^^^^^^^^^^^^^^^^^^^^

            - Improved :doc:`bugprone-easily-swappable-parameters
              <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by
              correcting a spelling mistake on its option
              ``NamePrefixSuffixSilenceDissimilarityTreshold``.

            - Improved :doc:`bugprone-exception-escape
              <clang-tidy/checks/bugprone/exception-escape>` check's handling of lambdas:
              exceptions from captures are now diagnosed, exceptions in the bodies of
              lambdas that aren't actually invoked are not.

            - Improved :doc:`bugprone-easily-swappable-parameters
              <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by
              correcting a spelling mistake on its option
              ``NamePrefixSuffixSilenceDissimilarityTreshold``.

            """
        )
        with tempfile.TemporaryDirectory() as td:
            rn_doc = os.path.join(td, "ReleaseNotes.rst")
            out_path = os.path.join(td, "out.rst")
            with open(rn_doc, "w", encoding="utf-8") as f:
                f.write(rn_text)

            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = _mod.process_release_notes(out_path, rn_doc)
            self.assertEqual(rc, 0)
            self.assertIn(
                "Entries in 'clang-tools-extra/docs/ReleaseNotes.rst' are not alphabetically sorted.",
                buf.getvalue(),
            )

            with open(out_path, "r", encoding="utf-8") as f:
                out = f.read()
            expected_out = textwrap.dedent(
                """\
                Changes in existing checks
                ^^^^^^^^^^^^^^^^^^^^^^^^^^

                - Improved :doc:`bugprone-easily-swappable-parameters
                  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by
                  correcting a spelling mistake on its option
                  ``NamePrefixSuffixSilenceDissimilarityTreshold``.

                - Improved :doc:`bugprone-easily-swappable-parameters
                  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by
                  correcting a spelling mistake on its option
                  ``NamePrefixSuffixSilenceDissimilarityTreshold``.

                - Improved :doc:`bugprone-exception-escape
                  <clang-tidy/checks/bugprone/exception-escape>` check's handling of lambdas:
                  exceptions from captures are now diagnosed, exceptions in the bodies of
                  lambdas that aren't actually invoked are not.


                """
            )
            self.assertEqual(out, expected_out)

    def test_process_release_notes_with_duplicates_fails(self) -> None:
        # Sorting is already correct but duplicates exist, should return 3 and report.
        rn_text = textwrap.dedent(
            """\
            Changes in existing checks
            ^^^^^^^^^^^^^^^^^^^^^^^^^^

            - Improved :doc:`bugprone-easily-swappable-parameters
              <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by
              correcting a spelling mistake on its option
              ``NamePrefixSuffixSilenceDissimilarityTreshold``.

            - Improved :doc:`bugprone-easily-swappable-parameters
              <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by
              correcting a spelling mistake on its option
              ``NamePrefixSuffixSilenceDissimilarityTreshold``.

            - Improved :doc:`bugprone-exception-escape
              <clang-tidy/checks/bugprone/exception-escape>` check's handling of lambdas:
              exceptions from captures are now diagnosed, exceptions in the bodies of
              lambdas that aren't actually invoked are not.

            """
        )
        with tempfile.TemporaryDirectory() as td:
            rn_doc = os.path.join(td, "ReleaseNotes.rst")
            out_path = os.path.join(td, "out.rst")
            with open(rn_doc, "w", encoding="utf-8") as f:
                f.write(rn_text)

            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = _mod.process_release_notes(out_path, rn_doc)

            self.assertEqual(rc, 3)
            expected_report = textwrap.dedent(
                """\
                Error: Duplicate entries in 'Changes in existing checks'.

                Please merge these entries into a single bullet point.

                -- Duplicate: - Improved :doc:`bugprone-easily-swappable-parameters

                - At line 4:
                - Improved :doc:`bugprone-easily-swappable-parameters
                  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by
                  correcting a spelling mistake on its option
                  ``NamePrefixSuffixSilenceDissimilarityTreshold``.

                - At line 9:
                - Improved :doc:`bugprone-easily-swappable-parameters
                  <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by
                  correcting a spelling mistake on its option
                  ``NamePrefixSuffixSilenceDissimilarityTreshold``.

                """
            )
            self.assertEqual(buf.getvalue(), expected_report)

            with open(out_path, "r", encoding="utf-8") as f:
                out = f.read()
            self.assertEqual(out, rn_text)

    def test_release_notes_handles_nested_sub_bullets(self) -> None:
        rn_text = textwrap.dedent(
            """\
            Changes in existing checks
            ^^^^^^^^^^^^^^^^^^^^^^^^^^

            - Improved :doc:`bugprone-easily-swappable-parameters
              <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by
              correcting a spelling mistake on its option
              ``NamePrefixSuffixSilenceDissimilarityTreshold``.

            - Improved :doc:`llvm-prefer-isa-or-dyn-cast-in-conditionals
              <clang-tidy/checks/llvm/prefer-isa-or-dyn-cast-in-conditionals>` check:

              - Fix-it handles callees with nested-name-specifier correctly.

              - ``if`` statements with init-statement (``if (auto X = ...; ...)``) are
                handled correctly.

              - ``for`` loops are supported.

            - Improved :doc:`bugprone-exception-escape
              <clang-tidy/checks/bugprone/exception-escape>` check's handling of lambdas:
              exceptions from captures are now diagnosed, exceptions in the bodies of
              lambdas that aren't actually invoked are not.

            """
        )

        out = _mod.normalize_release_notes(rn_text.splitlines(True))

        expected_out = textwrap.dedent(
            """\
            Changes in existing checks
            ^^^^^^^^^^^^^^^^^^^^^^^^^^

            - Improved :doc:`bugprone-easily-swappable-parameters
              <clang-tidy/checks/bugprone/easily-swappable-parameters>` check by
              correcting a spelling mistake on its option
              ``NamePrefixSuffixSilenceDissimilarityTreshold``.

            - Improved :doc:`bugprone-exception-escape
              <clang-tidy/checks/bugprone/exception-escape>` check's handling of lambdas:
              exceptions from captures are now diagnosed, exceptions in the bodies of
              lambdas that aren't actually invoked are not.

            - Improved :doc:`llvm-prefer-isa-or-dyn-cast-in-conditionals
              <clang-tidy/checks/llvm/prefer-isa-or-dyn-cast-in-conditionals>` check:

              - Fix-it handles callees with nested-name-specifier correctly.

              - ``if`` statements with init-statement (``if (auto X = ...; ...)``) are
                handled correctly.

              - ``for`` loops are supported.


           """
        )
        self.assertEqual(out, expected_out)

    def test_process_checks_list_normalizes_output(self) -> None:
        list_text = textwrap.dedent(
            """\
            .. csv-table:: List
               :header: "Name", "Redirect", "Offers fixes"

               :doc:`cert-dcl16-c <cert/dcl16-c>`, :doc:`readability-uppercase-literal-suffix <readability/uppercase-literal-suffix>`, "Yes"
               :doc:`cert-con36-c <cert/con36-c>`, :doc:`bugprone-spuriously-wake-up-functions <bugprone/spuriously-wake-up-functions>`,
               :doc:`cert-dcl37-c <cert/dcl37-c>`, :doc:`bugprone-reserved-identifier <bugprone/reserved-identifier>`, "Yes"
               :doc:`cert-arr39-c <cert/arr39-c>`, :doc:`bugprone-sizeof-expression <bugprone/sizeof-expression>`,
            """
        )
        with tempfile.TemporaryDirectory() as td:
            in_doc = os.path.join(td, "list.rst")
            out_doc = os.path.join(td, "out.rst")
            with open(in_doc, "w", encoding="utf-8") as f:
                f.write(list_text)
            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = _mod.process_checks_list(out_doc, in_doc)
            self.assertEqual(rc, 0)
            self.assertIn(
                "Checks in 'clang-tools-extra/docs/clang-tidy/checks/list.rst' csv-table are not alphabetically sorted.",
                buf.getvalue(),
            )
            self.assertEqual(rc, 0)
            with open(out_doc, "r", encoding="utf-8") as f:
                out = f.read()

            expected_out = textwrap.dedent(
                """\
                .. csv-table:: List
                   :header: "Name", "Redirect", "Offers fixes"

                   :doc:`cert-arr39-c <cert/arr39-c>`, :doc:`bugprone-sizeof-expression <bugprone/sizeof-expression>`,
                   :doc:`cert-con36-c <cert/con36-c>`, :doc:`bugprone-spuriously-wake-up-functions <bugprone/spuriously-wake-up-functions>`,
                   :doc:`cert-dcl16-c <cert/dcl16-c>`, :doc:`readability-uppercase-literal-suffix <readability/uppercase-literal-suffix>`, "Yes"
                   :doc:`cert-dcl37-c <cert/dcl37-c>`, :doc:`bugprone-reserved-identifier <bugprone/reserved-identifier>`, "Yes"
                """
            )
            self.assertEqual(out, expected_out)


if __name__ == "__main__":
    unittest.main()

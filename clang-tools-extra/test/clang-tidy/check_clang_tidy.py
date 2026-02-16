#!/usr/bin/env python3
#
# ===-----------------------------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===------------------------------------------------------------------------===#

"""
ClangTidy Test Helper
=====================

This script is used to simplify writing, running, and debugging tests compatible
with llvm-lit. By default it runs clang-tidy in fix mode and uses FileCheck to
verify messages and/or fixes.

For debugging, with --export-fixes, the tool simply exports fixes to a provided
file and does not run FileCheck.

Extra arguments, those after the first -- if any, are passed to either
clang-tidy or clang:
* Arguments between the first -- and second -- are clang-tidy arguments.
  * May be only whitespace if there are no clang-tidy arguments.
  * clang-tidy's --config would go here.
* Arguments after the second -- are clang arguments

Examples
--------

  // RUN: %check_clang_tidy %s llvm-include-order %t -- -- -isystem %S/Inputs

or

  // RUN: %check_clang_tidy %s llvm-include-order --export-fixes=fixes.yaml %t -std=c++20

Notes
-----
  -std=c++(98|11|14|17|20)-or-later:
    This flag will cause multiple runs within the same check_clang_tidy
    execution. Make sure you don't have shared state across these runs.
"""

import argparse
import os
import pathlib
import platform
import re
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


def write_file(file_name: str, text: str) -> None:
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(text)
        f.truncate()


def try_run(args: List[str], raise_error: bool = True) -> str:
    try:
        process_output = subprocess.check_output(args, stderr=subprocess.STDOUT).decode(
            errors="ignore"
        )
    except subprocess.CalledProcessError as e:
        process_output = e.output.decode(errors="ignore")
        print("%s failed:\n%s" % (" ".join(args), process_output))
        if raise_error:
            raise
    return process_output


# This class represents the appearance of a message prefix in a file.
class MessagePrefix:
    def __init__(self, label: str) -> None:
        self.has_message = False
        self.prefixes: List[str] = []
        self.label = label

    def check(self, file_check_suffix: str, input_text: str) -> bool:
        self.prefix = self.label + file_check_suffix
        self.has_message = self.prefix in input_text
        if self.has_message:
            self.prefixes.append(self.prefix)
        return self.has_message


class CheckRunner:
    def __init__(self, args: argparse.Namespace, extra_args: List[str]) -> None:
        self.resource_dir = args.resource_dir
        self.assume_file_name = args.assume_filename
        self.input_file_name = args.input_file_name
        self.check_name = args.check_name
        self.temp_file_name = args.temp_file_name
        self.check_headers = args.check_headers
        self.original_file_name = f"{self.temp_file_name}.orig"
        self.expect_clang_tidy_error = args.expect_clang_tidy_error
        self.std = args.std
        self.check_suffix = args.check_suffix
        self.input_text = ""
        self.has_check_fixes = False
        self.has_check_messages = False
        self.has_check_notes = False
        self.expect_no_diagnosis = False
        self.export_fixes = args.export_fixes
        self.fixes = MessagePrefix("CHECK-FIXES")
        self.messages = MessagePrefix("CHECK-MESSAGES")
        self.notes = MessagePrefix("CHECK-NOTES")
        self.match_partial_fixes = args.match_partial_fixes

        file_name_with_extension = self.assume_file_name or self.input_file_name
        _, extension = os.path.splitext(file_name_with_extension)
        if extension not in [".c", ".hpp", ".m", ".mm", ".cu"]:
            extension = ".cpp"
        self.temp_file_name = self.temp_file_name + extension

        self.clang_extra_args = []
        self.clang_tidy_extra_args = extra_args
        if "--" in extra_args:
            i = self.clang_tidy_extra_args.index("--")
            self.clang_extra_args = self.clang_tidy_extra_args[i + 1 :]
            self.clang_tidy_extra_args = self.clang_tidy_extra_args[:i]

        self.check_header_map: Dict[str, str] = {}
        self.header_dir = f"{self.temp_file_name}.headers"
        if self.check_headers:
            self.check_header_map = {
                os.path.normcase(
                    os.path.abspath(os.path.join(self.header_dir, os.path.basename(h)))
                ): os.path.abspath(h)
                for h in self.check_headers
            }

            self.clang_extra_args.insert(0, f"-I{self.header_dir}")

        # If the test does not specify a config style, force an empty one; otherwise
        # auto-detection logic can discover a ".clang-tidy" file that is not related to
        # the test.
        if not any(
            [re.match("^-?-config(-file)?=", arg) for arg in self.clang_tidy_extra_args]
        ):
            self.clang_tidy_extra_args.append("--config={}")

        if extension in [".m", ".mm"]:
            self.clang_extra_args = [
                "-fobjc-abi-version=2",
                "-fobjc-arc",
                "-fblocks",
            ] + self.clang_extra_args

        self.clang_extra_args.append(f"-std={self.std}")

        # Tests should not rely on STL being available, and instead provide mock
        # implementations of relevant APIs.
        self.clang_extra_args.append("-nostdinc++")

        if self.resource_dir is not None:
            self.clang_extra_args.append("-resource-dir=%s" % self.resource_dir)

    def read_input(self) -> None:
        # Use a "\\?\" prefix on Windows to handle long file paths transparently:
        # https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation
        file_name = self.input_file_name
        if platform.system() == "Windows":
            file_name = "\\\\?\\" + os.path.abspath(file_name)
        with open(file_name, "r", encoding="utf-8") as input_file:
            self.input_text = input_file.read()

    def get_prefixes(self) -> None:
        for suffix in self.check_suffix:
            if suffix and not re.match("^[A-Z0-9\\-]+$", suffix):
                sys.exit(
                    'Only A..Z, 0..9 and "-" are allowed in check suffixes list, '
                    f'but "{suffix}" was given'
                )

            file_check_suffix = ("-" + suffix) if suffix else ""

            has_check_fix = self.fixes.check(file_check_suffix, self.input_text)
            self.has_check_fixes = self.has_check_fixes or has_check_fix

            has_check_message = self.messages.check(file_check_suffix, self.input_text)
            self.has_check_messages = self.has_check_messages or has_check_message

            has_check_note = self.notes.check(file_check_suffix, self.input_text)
            self.has_check_notes = self.has_check_notes or has_check_note

            if has_check_note and has_check_message:
                sys.exit(
                    "Please use either %s or %s but not both"
                    % (self.notes.prefix, self.messages.prefix)
                )

            if not has_check_fix and not has_check_message and not has_check_note:
                self.expect_no_diagnosis = True

        expect_diagnosis = (
            self.has_check_fixes or self.has_check_messages or self.has_check_notes
        )
        if self.expect_no_diagnosis and expect_diagnosis:
            sys.exit(
                "%s, %s or %s not found in the input"
                % (
                    self.fixes.prefix,
                    self.messages.prefix,
                    self.notes.prefix,
                )
            )
        assert expect_diagnosis or self.expect_no_diagnosis

    def _remove_filecheck_content(self, text: str) -> str:
        # Remove the contents of the CHECK lines to avoid CHECKs matching on
        # themselves. We need to keep the comments to preserve line numbers while
        # avoiding empty lines which could potentially trigger formatting-related
        # checks.
        return re.sub("// *CHECK-[A-Z0-9\\-]*:[^\r\n]*", "//", text)

    def _filter_prefixes(self, prefixes: Sequence[str], check_file: str) -> List[str]:
        """
        Filter prefixes to only those present in the check file.

        - Input:
            prefixes: A list of potential FileCheck prefixes.
            check_file: The file to check for prefix presence.
        - Output:
            A list of prefixes found in the check file.

        FileCheck fails if a specified prefix is not present. This is common
        in header testing scenarios where expectations differ between the
        main file and the header (e.g. the main file might verify code
        changes while the header only verifies warnings).
        """
        if check_file == self.input_file_name:
            content = self.input_text
        else:
            with open(check_file, "r", encoding="utf-8") as f:
                content = f.read()
        return [p for p in prefixes if p in content]

    def prepare_test_inputs(self) -> None:
        cleaned_test = self._remove_filecheck_content(self.input_text)
        write_file(self.temp_file_name, cleaned_test)
        write_file(self.original_file_name, cleaned_test)

        if self.check_headers:
            os.makedirs(self.header_dir, exist_ok=True)

            for temp_header_path, header in self.check_header_map.items():
                with open(header, "r", encoding="utf-8") as f:
                    cleaned_header = self._remove_filecheck_content(f.read())

                write_file(temp_header_path, cleaned_header)
                write_file(f"{temp_header_path}.orig", cleaned_header)

    def run_clang_tidy(self) -> str:
        args = (
            [
                "clang-tidy",
                "--experimental-custom-checks",
                self.temp_file_name,
            ]
            + [
                (
                    "-fix"
                    if self.export_fixes is None
                    else f"--export-fixes={self.export_fixes}"
                )
            ]
            + [
                f"--checks=-*,{self.check_name}",
            ]
            + self.clang_tidy_extra_args
            + ["--"]
            + self.clang_extra_args
        )
        if self.expect_clang_tidy_error:
            args.insert(0, "not")
        print(f"Running {repr(args)}...")
        clang_tidy_output = try_run(args)
        print("------------------------ clang-tidy output -----------------------")
        print(
            clang_tidy_output.encode(sys.stdout.encoding, errors="replace").decode(
                sys.stdout.encoding
            )
        )
        print("------------------------------------------------------------------")

        diff_output = try_run(
            ["diff", "-u", self.original_file_name, self.temp_file_name], False
        )
        if self.check_headers:
            for temp_header_path in self.check_header_map:
                diff_output += try_run(
                    ["diff", "-u", f"{temp_header_path}.orig", temp_header_path], False
                )

        print("------------------------------ Fixes -----------------------------")
        print(diff_output)
        print("------------------------------------------------------------------")
        return clang_tidy_output

    def check_no_diagnosis(self, clang_tidy_output: str) -> None:
        if clang_tidy_output != "":
            sys.exit("No diagnostics were expected, but found the ones above")

    def check_fixes(self, input_file: str = "", check_file: str = "") -> None:
        if not (check_file or self.has_check_fixes):
            return

        input_file = input_file or self.temp_file_name
        check_file = check_file or self.input_file_name
        active_prefixes = self._filter_prefixes(self.fixes.prefixes, check_file)

        if not active_prefixes:
            return

        try_run(
            [
                "FileCheck",
                f"--input-file={input_file}",
                check_file,
                f"--check-prefixes={','.join(active_prefixes)}",
                (
                    "--strict-whitespace"  # Keeping past behavior.
                    if self.match_partial_fixes
                    else "--match-full-lines"
                ),
            ]
        )

    def check_messages(
        self,
        clang_tidy_output: str,
        messages_file: str = "",
        check_file: str = "",
    ) -> None:
        if not check_file and not self.has_check_messages:
            return

        messages_file = messages_file or f"{self.temp_file_name}.msg"
        check_file = check_file or self.input_file_name

        active_prefixes = self._filter_prefixes(self.messages.prefixes, check_file)
        if not active_prefixes:
            return

        write_file(messages_file, clang_tidy_output)
        try_run(
            [
                "FileCheck",
                f"-input-file={messages_file}",
                check_file,
                f"-check-prefixes={','.join(active_prefixes)}",
                "-implicit-check-not={{warning|error}}:",
            ]
        )

    def check_notes(self, clang_tidy_output: str) -> None:
        if self.has_check_notes:
            notes_file = f"{self.temp_file_name}.notes"
            filtered_output = [
                line
                for line in clang_tidy_output.splitlines()
                if not ("note: FIX-IT applied" in line)
            ]
            write_file(notes_file, "\n".join(filtered_output))
            try_run(
                [
                    "FileCheck",
                    f"-input-file={notes_file}",
                    self.input_file_name,
                    f"-check-prefixes={','.join(self.notes.prefixes)}",
                    "-implicit-check-not={{note|warning|error}}:",
                ]
            )

    def _separate_messages(self, clang_tidy_output: str) -> Tuple[str, Dict[str, str]]:
        """
        Separates diagnostics for the main file and headers.

        - Input: The raw diagnostic output from clang-tidy.
        - Output: A tuple containing:
            1. The diagnostic output for the main file.
            2. A dictionary mapping header files to their diagnostics.
        """
        if not self.check_headers:
            return clang_tidy_output, {}

        header_messages = defaultdict(list)
        remaining_lines: List[str] = []
        current_file = ""

        for line in clang_tidy_output.splitlines(keepends=True):
            if re.match(r"^\d+ warnings? generated\.", line):
                continue

            # Matches the beginning of a clang-tidy diagnostic line,
            # which starts with "file_path:line:col: ".
            if match := re.match(r"^(.+):\d+:\d+: ", line):
                abs_path = os.path.normcase(os.path.abspath(match.group(1)))
                current_file = abs_path if abs_path in self.check_header_map else ""

            dest_list = (
                header_messages[current_file] if current_file else remaining_lines
            )
            dest_list.append(line)

        header_messages_str = {k: "".join(v) for k, v in header_messages.items()}
        return "".join(remaining_lines), header_messages_str

    def run(self) -> None:
        self.read_input()
        if self.export_fixes is None:
            self.get_prefixes()
        self.prepare_test_inputs()
        clang_tidy_output = self.run_clang_tidy()
        main_output, header_messages = self._separate_messages(clang_tidy_output)

        if self.expect_no_diagnosis:
            self.check_no_diagnosis(main_output)
        elif self.export_fixes is None:
            self.check_fixes()
            self.check_messages(main_output)
            self.check_notes(main_output)

            for temp_header, original_header in self.check_header_map.items():
                self.check_fixes(
                    input_file=temp_header,
                    check_file=original_header,
                )

                if temp_header in header_messages:
                    self.check_messages(
                        header_messages[temp_header],
                        messages_file=f"{temp_header}.msg",
                        check_file=original_header,
                    )


CPP_STANDARDS = [
    "c++98",
    "c++11",
    ("c++14", "c++1y"),
    ("c++17", "c++1z"),
    ("c++20", "c++2a"),
    ("c++23", "c++2b"),
    ("c++26", "c++2c"),
]
C_STANDARDS = ["c99", ("c11", "c1x"), "c17", ("c23", "c2x"), "c2y"]


def expand_std(std: str) -> List[str]:
    split_std, or_later, _ = std.partition("-or-later")

    if not or_later:
        return [split_std]

    for standard_list in (CPP_STANDARDS, C_STANDARDS):
        item = next(
            (
                i
                for i, v in enumerate(standard_list)
                if (split_std in v if isinstance(v, (list, tuple)) else split_std == v)
            ),
            None,
        )
        if item is not None:
            return [split_std] + [
                x if isinstance(x, str) else x[0] for x in standard_list[item + 1 :]
            ]
    return [std]


def csv(string: str) -> List[str]:
    return string.split(",")


def parse_arguments() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).stem,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-expect-clang-tidy-error", action="store_true")
    parser.add_argument("-resource-dir")
    parser.add_argument("-assume-filename")
    parser.add_argument("input_file_name")
    parser.add_argument("check_name")
    parser.add_argument("temp_file_name")
    parser.add_argument(
        "-check-suffix",
        "-check-suffixes",
        default=[""],
        type=csv,
        help="comma-separated list of FileCheck suffixes",
    )
    parser.add_argument(
        "-check-header",
        action="append",
        dest="check_headers",
        default=[],
        help="Header files to check",
    )
    parser.add_argument(
        "-export-fixes",
        default=None,
        type=str,
        metavar="file",
        help="A file to export fixes into instead of fixing.",
    )
    parser.add_argument(
        "-std",
        type=csv,
        default=None,
        help="Passed to clang. Special -or-later values are expanded.",
    )
    parser.add_argument(
        "--match-partial-fixes",
        action="store_true",
        help="allow partial line matches for fixes",
    )

    args, extra_args = parser.parse_known_args()
    if args.std is None:
        _, extension = os.path.splitext(args.assume_filename or args.input_file_name)
        args.std = ["c99-or-later" if extension in [".c", ".m"] else "c++11-or-later"]

    return (args, extra_args)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    args, extra_args = parse_arguments()

    abbreviated_stds = args.std
    for abbreviated_std in abbreviated_stds:
        for std in expand_std(abbreviated_std):
            args.std = std
            CheckRunner(args, extra_args).run()


if __name__ == "__main__":
    main()

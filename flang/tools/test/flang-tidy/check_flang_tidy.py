#!/usr/bin/env python3
#
# ===- check_flang_tidy.py - FlangTidy Test Helper ------------*- python -*--===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===------------------------------------------------------------------------===#

"""
FlangTidy Test Helper
=====================

This script helps run flang-tidy checks with llvm-lit. By default, it runs flang-tidy
without applying fixes and uses FileCheck to verify warnings.
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys


def write_file(file_name, text):
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(text)
        f.truncate()


def try_run(args, raise_error=True):
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


class MessagePrefix:
    def __init__(self, label):
        self.has_message = False
        self.prefixes = []
        self.label = label

    def check(self, file_check_suffix, input_text):
        self.prefix = self.label + file_check_suffix
        self.has_message = self.prefix in input_text
        if self.has_message:
            self.prefixes.append(self.prefix)
        return self.has_message


class CheckRunner:
    def __init__(self, args, extra_args):
        self.resource_dir = args.resource_dir
        self.input_file_name = args.input_file_name
        self.check_name = args.check_name
        self.temp_file_name = args.temp_file_name
        self.expect_flang_tidy_error = args.expect_flang_tidy_error
        self.std = args.std
        self.check_suffix = args.check_suffix
        self.input_text = ""
        self.has_check_messages = False
        self.expect_no_diagnosis = False
        self.messages = MessagePrefix("CHECK-MESSAGES")

        file_name_with_extension = self.input_file_name
        _, extension = os.path.splitext(file_name_with_extension)
        if extension not in [".f", ".f90", ".f95"]:
            extension = ".f90"
        self.temp_file_name = self.temp_file_name + extension

        self.extra_args = extra_args
        self.flang_extra_args = []

        if "--" in extra_args:
            i = self.extra_args.index("--")
            self.flang_extra_args = self.extra_args[i + 1 :]
            self.extra_args = self.extra_args[:i]

    def read_input(self):
        with open(self.input_file_name, "r", encoding="utf-8") as input_file:
            self.input_text = input_file.read()

    def get_prefixes(self):
        for suffix in self.check_suffix:
            if suffix and not re.match("^[A-Z0-9\\-]+$", suffix):
                sys.exit(
                    'Only A..Z, 0..9 and "-" are allowed in check suffixes list,'
                    + ' but "%s" was given' % suffix
                )

            file_check_suffix = ("-" + suffix) if suffix else ""

            has_check_message = self.messages.check(file_check_suffix, self.input_text)
            self.has_check_messages = self.has_check_messages or has_check_message

            if not has_check_message:
                self.expect_no_diagnosis = True

        if self.expect_no_diagnosis and self.has_check_messages:
            sys.exit(
                "%s not found in the input" % self.messages.prefix
            )
        assert self.has_check_messages or self.expect_no_diagnosis

    def prepare_test_inputs(self):
        cleaned_test = re.sub("// *CHECK-[A-Z0-9\\-]*:[^\r\n]*", "//", self.input_text)
        write_file(self.temp_file_name, cleaned_test)

    def run_flang_tidy(self):
        args = (
            [
                "flang-tidy",
                self.temp_file_name,
            ]
            + ["--checks=" + self.check_name]
            + self.extra_args
            #+ ["--"]
            #+ self.flang_extra_args
        )
        if self.expect_flang_tidy_error:
            args.insert(0, "not")
        print("Running " + repr(args) + "...")
        flang_tidy_output = try_run(args)
        print("------------------------ flang-tidy output -----------------------")
        print(
            flang_tidy_output.encode(sys.stdout.encoding, errors="replace").decode(
                sys.stdout.encoding
            )
        )
        print("------------------------------------------------------------------")
        return flang_tidy_output

    def check_messages(self, flang_tidy_output):
        if self.has_check_messages:
            messages_file = self.temp_file_name + ".msg"
            write_file(messages_file, flang_tidy_output)
            try_run(
                [
                    "FileCheck",
                    "-input-file=" + messages_file,
                    self.input_file_name,
                    "-check-prefixes=" + ",".join(self.messages.prefixes),
                    "-implicit-check-not={{warning|error}}:",
                ]
            )

    def run(self):
        self.read_input()
        self.get_prefixes()
        self.prepare_test_inputs()
        flang_tidy_output = self.run_flang_tidy()
        if self.expect_no_diagnosis:
            if flang_tidy_output != "":
                sys.exit("No diagnostics were expected, but found the ones above")
        else:
            self.check_messages(flang_tidy_output)


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).stem,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-expect-flang-tidy-error", action="store_true")
    parser.add_argument("-resource-dir")
    parser.add_argument("input_file_name")
    parser.add_argument("check_name")
    parser.add_argument("temp_file_name")
    parser.add_argument(
        "-check-suffix",
        "-check-suffixes",
        default=[""],
        type=lambda x: x.split(","),
        help="comma-separated list of FileCheck suffixes",
    )
    parser.add_argument(
        "-std",
        default="f2003",
        help="Fortran standard to pass to flang.",
    )
    return parser.parse_known_args()


def main():
    args, extra_args = parse_arguments()
    CheckRunner(args, extra_args).run()


if __name__ == "__main__":
    main()
#!/usr/bin/env python
#
#===- check-analyzer-fixit.py - Static Analyzer test helper ---*- python -*-===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#
#
# This file copy-pasted mostly from the Clang-Tidy's 'check_clang_tidy.py'.
#
#===------------------------------------------------------------------------===#

r"""
Clang Static Analyzer test helper
=================================

This script runs the Analyzer in fix-it mode and verify fixes, warnings, notes.

Usage:
  check-analyzer-fixit.py <source-file> <temp-file> [analyzer arguments]

Example:
  // RUN: %check-analyzer-fixit %s %t -analyzer-checker=core
"""

import argparse
import os
import re
import subprocess
import sys


def write_file(file_name, text):
    with open(file_name, 'w') as f:
        f.write(text)


def run_test_once(args, extra_args):
    input_file_name = args.input_file_name
    temp_file_name = args.temp_file_name
    clang_analyzer_extra_args = extra_args

    file_name_with_extension = input_file_name
    _, extension = os.path.splitext(file_name_with_extension)
    if extension not in ['.c', '.hpp', '.m', '.mm']:
        extension = '.cpp'
    temp_file_name = temp_file_name + extension

    with open(input_file_name, 'r') as input_file:
        input_text = input_file.read()

    # Remove the contents of the CHECK lines to avoid CHECKs matching on
    # themselves.  We need to keep the comments to preserve line numbers while
    # avoiding empty lines which could potentially trigger formatting-related
    # checks.
    cleaned_test = re.sub('// *CHECK-[A-Z0-9\-]*:[^\r\n]*', '//', input_text)
    write_file(temp_file_name, cleaned_test)

    original_file_name = temp_file_name + ".orig"
    write_file(original_file_name, cleaned_test)

    try:
        builtin_include_dir = subprocess.check_output(
            ['clang', '-print-file-name=include'], stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        print('Cannot print Clang include directory: ' + e.output.decode())

    builtin_include_dir = os.path.normpath(builtin_include_dir)

    args = (['clang', '-cc1', '-internal-isystem', builtin_include_dir,
             '-nostdsysteminc', '-analyze', '-analyzer-constraints=range',
             '-analyzer-config', 'apply-fixits=true']
            + clang_analyzer_extra_args + ['-verify', temp_file_name])

    print('Running ' + str(args) + '...')

    try:
        clang_analyzer_output = \
            subprocess.check_output(args, stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        print('Clang Static Analyzer test failed:\n' + e.output.decode())
        raise

    print('----------------- Clang Static Analyzer output -----------------\n' +
          clang_analyzer_output +
          '\n--------------------------------------------------------------')

    try:
        diff_output = subprocess.check_output(
            ['diff', '-u', original_file_name, temp_file_name],
            stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        diff_output = e.output

    print('----------------------------- Fixes ----------------------------\n' +
          diff_output.decode() +
          '\n--------------------------------------------------------------')

    try:
        subprocess.check_output(
            ['FileCheck', '-input-file=' + temp_file_name, input_file_name,
             '-check-prefixes=CHECK-FIXES', '-strict-whitespace'],
            stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print('FileCheck failed:\n' + e.output.decode())
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file_name')
    parser.add_argument('temp_file_name')

    args, extra_args = parser.parse_known_args()
    run_test_once(args, extra_args)


if __name__ == '__main__':
    main()

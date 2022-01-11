#!/usr/bin/env python
#
#===- check_clang_tidy.py - ClangTidy Test Helper ------------*- python -*--===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

r"""
ClangTidy Test Helper
=====================

This script runs clang-tidy in fix mode and verify fixes, messages or both.

Usage:
  check_clang_tidy.py [-resource-dir=<resource-dir>] \
    [-assume-filename=<file-with-source-extension>] \
    [-check-suffix=<comma-separated-file-check-suffixes>] \
    [-check-suffixes=<comma-separated-file-check-suffixes>] \
    <source-file> <check-name> <temp-file> \
    -- [optional clang-tidy arguments]

Example:
  // RUN: %check_clang_tidy %s llvm-include-order %t -- -- -isystem %S/Inputs
"""

import argparse
import os
import re
import subprocess
import sys


def write_file(file_name, text):
  with open(file_name, 'w', encoding='utf-8') as f:
    f.write(text)
    f.truncate()


def try_run(args, raise_error=True):
  try:
    process_output = \
      subprocess.check_output(args, stderr=subprocess.STDOUT).decode(errors='ignore')
  except subprocess.CalledProcessError as e:
    process_output = e.output.decode(errors='ignore')
    print('%s failed:\n%s' % (' '.join(args), process_output))
    if raise_error:
      raise
  return process_output


class CheckRunner:
  def __init__(self, args, extra_args):
    self.resource_dir = args.resource_dir
    self.assume_file_name = args.assume_filename
    self.input_file_name = args.input_file_name
    self.check_name = args.check_name
    self.temp_file_name = args.temp_file_name
    self.original_file_name = self.temp_file_name + ".orig"
    self.expect_clang_tidy_error = args.expect_clang_tidy_error
    self.std = args.std
    self.check_suffix = args.check_suffix
    self.input_text = ''
    self.check_fixes_prefixes = []
    self.check_messages_prefixes = []
    self.check_notes_prefixes = []
    self.has_check_fixes = False
    self.has_check_messages = False
    self.has_check_notes = False

    file_name_with_extension = self.assume_file_name or self.input_file_name
    _, extension = os.path.splitext(file_name_with_extension)
    if extension not in ['.c', '.hpp', '.m', '.mm']:
      extension = '.cpp'
    self.temp_file_name = self.temp_file_name + extension

    self.clang_extra_args = []
    self.clang_tidy_extra_args = extra_args
    if '--' in extra_args:
      i = self.clang_tidy_extra_args.index('--')
      self.clang_extra_args = self.clang_tidy_extra_args[i + 1:]
      self.clang_tidy_extra_args = self.clang_tidy_extra_args[:i]

    # If the test does not specify a config style, force an empty one; otherwise
    # auto-detection logic can discover a ".clang-tidy" file that is not related to
    # the test.
    if not any(
        [arg.startswith('-config=') for arg in self.clang_tidy_extra_args]):
      self.clang_tidy_extra_args.append('-config={}')

    if extension in ['.m', '.mm']:
      self.clang_extra_args = ['-fobjc-abi-version=2', '-fobjc-arc', '-fblocks'] + \
          self.clang_extra_args

    if extension in ['.cpp', '.hpp', '.mm']:
      self.clang_extra_args.append('-std=' + self.std)

    # Tests should not rely on STL being available, and instead provide mock
    # implementations of relevant APIs.
    self.clang_extra_args.append('-nostdinc++')

    if self.resource_dir is not None:
      self.clang_extra_args.append('-resource-dir=%s' % self.resource_dir)

  def read_input(self):
    with open(self.input_file_name, 'r', encoding='utf-8') as input_file:
      self.input_text = input_file.read()

  def get_prefixes(self):
    for check in self.check_suffix:
      if check and not re.match('^[A-Z0-9\\-]+$', check):
        sys.exit('Only A..Z, 0..9 and "-" are allowed in check suffixes list,'
                 + ' but "%s" was given' % check)

      file_check_suffix = ('-' + check) if check else ''
      check_fixes_prefix = 'CHECK-FIXES' + file_check_suffix
      check_messages_prefix = 'CHECK-MESSAGES' + file_check_suffix
      check_notes_prefix = 'CHECK-NOTES' + file_check_suffix

      has_check_fix = check_fixes_prefix in self.input_text
      has_check_message = check_messages_prefix in self.input_text
      has_check_note = check_notes_prefix in self.input_text

      if has_check_note and has_check_message:
        sys.exit('Please use either %s or %s but not both' %
          (check_notes_prefix, check_messages_prefix))

      if not has_check_fix and not has_check_message and not has_check_note:
        sys.exit('%s, %s or %s not found in the input' %
          (check_fixes_prefix, check_messages_prefix, check_notes_prefix))

      self.has_check_fixes = self.has_check_fixes or has_check_fix
      self.has_check_messages = self.has_check_messages or has_check_message
      self.has_check_notes = self.has_check_notes or has_check_note

      if has_check_fix:
        self.check_fixes_prefixes.append(check_fixes_prefix)
      if has_check_message:
        self.check_messages_prefixes.append(check_messages_prefix)
      if has_check_note:
        self.check_notes_prefixes.append(check_notes_prefix)

    assert self.has_check_fixes or self.has_check_messages or self.has_check_notes

  def prepare_test_inputs(self):
    # Remove the contents of the CHECK lines to avoid CHECKs matching on
    # themselves.  We need to keep the comments to preserve line numbers while
    # avoiding empty lines which could potentially trigger formatting-related
    # checks.
    cleaned_test = re.sub('// *CHECK-[A-Z0-9\\-]*:[^\r\n]*', '//', self.input_text)
    write_file(self.temp_file_name, cleaned_test)
    write_file(self.original_file_name, cleaned_test)

  def run_clang_tidy(self):
    args = ['clang-tidy', self.temp_file_name, '-fix', '--checks=-*,' + self.check_name] + \
        self.clang_tidy_extra_args + ['--'] + self.clang_extra_args
    if self.expect_clang_tidy_error:
      args.insert(0, 'not')
    print('Running ' + repr(args) + '...')
    clang_tidy_output = try_run(args)
    print('------------------------ clang-tidy output -----------------------')
    print(clang_tidy_output.encode())
    print('\n------------------------------------------------------------------')

    diff_output = try_run(['diff', '-u', self.original_file_name, self.temp_file_name], False)
    print('------------------------------ Fixes -----------------------------\n' +
          diff_output +
          '\n------------------------------------------------------------------')
    return clang_tidy_output

  def check_fixes(self):
    if self.has_check_fixes:
      try_run(['FileCheck', '-input-file=' + self.temp_file_name, self.input_file_name,
              '-check-prefixes=' + ','.join(self.check_fixes_prefixes),
              '-strict-whitespace'])

  def check_messages(self, clang_tidy_output):
    if self.has_check_messages:
      messages_file = self.temp_file_name + '.msg'
      write_file(messages_file, clang_tidy_output)
      try_run(['FileCheck', '-input-file=' + messages_file, self.input_file_name,
             '-check-prefixes=' + ','.join(self.check_messages_prefixes),
             '-implicit-check-not={{warning|error}}:'])

  def check_notes(self, clang_tidy_output):
    if self.has_check_notes:
      notes_file = self.temp_file_name + '.notes'
      filtered_output = [line for line in clang_tidy_output.splitlines()
                         if not ("note: FIX-IT applied" in line)]
      write_file(notes_file, '\n'.join(filtered_output))
      try_run(['FileCheck', '-input-file=' + notes_file, self.input_file_name,
             '-check-prefixes=' + ','.join(self.check_notes_prefixes),
             '-implicit-check-not={{note|warning|error}}:'])

  def run(self):
    self.read_input()
    self.get_prefixes()
    self.prepare_test_inputs()
    clang_tidy_output = self.run_clang_tidy()
    self.check_fixes()
    self.check_messages(clang_tidy_output)
    self.check_notes(clang_tidy_output)


def expand_std(std):
  if std == 'c++98-or-later':
    return ['c++98', 'c++11', 'c++14', 'c++17', 'c++20']
  if std == 'c++11-or-later':
    return ['c++11', 'c++14', 'c++17', 'c++20']
  if std == 'c++14-or-later':
    return ['c++14', 'c++17', 'c++20']
  if std == 'c++17-or-later':
    return ['c++17', 'c++20']
  if std == 'c++20-or-later':
    return ['c++20']
  return [std]


def csv(string):
  return string.split(',')


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('-expect-clang-tidy-error', action='store_true')
  parser.add_argument('-resource-dir')
  parser.add_argument('-assume-filename')
  parser.add_argument('input_file_name')
  parser.add_argument('check_name')
  parser.add_argument('temp_file_name')
  parser.add_argument(
    '-check-suffix',
    '-check-suffixes',
    default=[''],
    type=csv,
    help='comma-separated list of FileCheck suffixes')
  parser.add_argument('-std', type=csv, default=['c++11-or-later'])
  return parser.parse_known_args()


def main():
  args, extra_args = parse_arguments()

  abbreviated_stds = args.std
  for abbreviated_std in abbreviated_stds:
    for std in expand_std(abbreviated_std):
      args.std = std
      CheckRunner(args, extra_args).run()


if __name__ == '__main__':
  main()

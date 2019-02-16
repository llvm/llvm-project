#!/usr/bin/env python
#
#===- check_clang_tidy.py - ClangTidy Test Helper ------------*- python -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
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
  with open(file_name, 'w') as f:
    f.write(text)
    f.truncate()

def csv(string):
  return string.split(',')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-expect-clang-tidy-error', action='store_true')
  parser.add_argument('-resource-dir')
  parser.add_argument('-assume-filename')
  parser.add_argument('input_file_name')
  parser.add_argument('check_name')
  parser.add_argument('temp_file_name')
  parser.add_argument('-check-suffix', '-check-suffixes',
                      default=[''], type=csv,
                      help="comma-separated list of FileCheck suffixes")

  args, extra_args = parser.parse_known_args()

  resource_dir = args.resource_dir
  assume_file_name = args.assume_filename
  input_file_name = args.input_file_name
  check_name = args.check_name
  temp_file_name = args.temp_file_name
  expect_clang_tidy_error = args.expect_clang_tidy_error

  file_name_with_extension = assume_file_name or input_file_name
  _, extension = os.path.splitext(file_name_with_extension)
  if extension not in ['.c', '.hpp', '.m', '.mm']:
    extension = '.cpp'
  temp_file_name = temp_file_name + extension

  clang_tidy_extra_args = extra_args
  if len(clang_tidy_extra_args) == 0:
    clang_tidy_extra_args = ['--']
    if extension in ['.cpp', '.hpp', '.mm']:
      clang_tidy_extra_args.append('--std=c++11')
    if extension in ['.m', '.mm']:
      clang_tidy_extra_args.extend(
          ['-fobjc-abi-version=2', '-fobjc-arc'])

  # Tests should not rely on STL being available, and instead provide mock
  # implementations of relevant APIs.
  clang_tidy_extra_args.append('-nostdinc++')

  if resource_dir is not None:
    clang_tidy_extra_args.append('-resource-dir=%s' % resource_dir)

  with open(input_file_name, 'r') as input_file:
    input_text = input_file.read()

  check_fixes_prefixes = []
  check_messages_prefixes = []
  check_notes_prefixes = []

  has_check_fixes = False
  has_check_messages = False
  has_check_notes = False

  for check in args.check_suffix:
    if check and not re.match('^[A-Z0-9\-]+$', check):
      sys.exit('Only A..Z, 0..9 and "-" are ' +
        'allowed in check suffixes list, but "%s" was given' % (check))

    file_check_suffix = ('-' + check) if check else ''
    check_fixes_prefix = 'CHECK-FIXES' + file_check_suffix
    check_messages_prefix = 'CHECK-MESSAGES' + file_check_suffix
    check_notes_prefix = 'CHECK-NOTES' + file_check_suffix

    has_check_fix = check_fixes_prefix in input_text
    has_check_message = check_messages_prefix in input_text
    has_check_note = check_notes_prefix in input_text

    if has_check_note and has_check_message:
      sys.exit('Please use either %s or %s but not both' %
        (check_notes_prefix, check_messages_prefix))

    if not has_check_fix and not has_check_message and not has_check_note:
      sys.exit('%s, %s or %s not found in the input' %
        (check_fixes_prefix, check_messages_prefix, check_notes_prefix))

    has_check_fixes = has_check_fixes or has_check_fix
    has_check_messages = has_check_messages or has_check_message
    has_check_notes = has_check_notes or has_check_note

    check_fixes_prefixes.append(check_fixes_prefix)
    check_messages_prefixes.append(check_messages_prefix)
    check_notes_prefixes.append(check_notes_prefix)

  assert has_check_fixes or has_check_messages or has_check_notes
  # Remove the contents of the CHECK lines to avoid CHECKs matching on
  # themselves.  We need to keep the comments to preserve line numbers while
  # avoiding empty lines which could potentially trigger formatting-related
  # checks.
  cleaned_test = re.sub('// *CHECK-[A-Z0-9\-]*:[^\r\n]*', '//', input_text)

  write_file(temp_file_name, cleaned_test)

  original_file_name = temp_file_name + ".orig"
  write_file(original_file_name, cleaned_test)

  args = ['clang-tidy', temp_file_name, '-fix', '--checks=-*,' + check_name] + \
        clang_tidy_extra_args
  if expect_clang_tidy_error:
    args.insert(0, 'not')
  print('Running ' + repr(args) + '...')
  try:
    clang_tidy_output = \
        subprocess.check_output(args, stderr=subprocess.STDOUT).decode()
  except subprocess.CalledProcessError as e:
    print('clang-tidy failed:\n' + e.output.decode())
    raise

  print('------------------------ clang-tidy output -----------------------\n' +
        clang_tidy_output +
        '\n------------------------------------------------------------------')

  try:
    diff_output = subprocess.check_output(
        ['diff', '-u', original_file_name, temp_file_name],
        stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError as e:
    diff_output = e.output

  print('------------------------------ Fixes -----------------------------\n' +
        diff_output.decode() +
        '\n------------------------------------------------------------------')

  if has_check_fixes:
    try:
      subprocess.check_output(
          ['FileCheck', '-input-file=' + temp_file_name, input_file_name,
           '-check-prefixes=' + ','.join(check_fixes_prefixes),
           '-strict-whitespace'],
          stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
      print('FileCheck failed:\n' + e.output.decode())
      raise

  if has_check_messages:
    messages_file = temp_file_name + '.msg'
    write_file(messages_file, clang_tidy_output)
    try:
      subprocess.check_output(
          ['FileCheck', '-input-file=' + messages_file, input_file_name,
           '-check-prefixes=' + ','.join(check_messages_prefixes),
           '-implicit-check-not={{warning|error}}:'],
          stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
      print('FileCheck failed:\n' + e.output.decode())
      raise

  if has_check_notes:
    notes_file = temp_file_name + '.notes'
    filtered_output = [line for line in clang_tidy_output.splitlines()
                       if not "note: FIX-IT applied" in line]
    write_file(notes_file, '\n'.join(filtered_output))
    try:
      subprocess.check_output(
          ['FileCheck', '-input-file=' + notes_file, input_file_name,
           '-check-prefixes=' + ','.join(check_notes_prefixes),
           '-implicit-check-not={{note|warning|error}}:'],
          stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
      print('FileCheck failed:\n' + e.output.decode())
      raise

if __name__ == '__main__':
  main()

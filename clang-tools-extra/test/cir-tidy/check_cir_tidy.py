#!/usr/bin/env python
#
#===- check_cir_tidy.py - CIRTidy Test Helper ------------*- python -*--=======#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

r"""
CIRTIDY Test Helper
=====================

This script runs cir-tidy and check outputed messages.

Usage:
  check_cir_tidy.py <source-file> <check-name> <temp-file> -- \
  [optional cir-tidy arguments]

Example:
  // RUN: %check_cir_tidy %s cir-lifetime-check %t --
"""

import argparse
import re
import subprocess
import sys
import shutil

def write_file(file_name, text):
  with open(file_name, 'w', encoding='utf-8') as f:
    f.write(text)
    f.truncate()


def run_test_once(args, extra_args):
  input_file_name = args.input_file_name
  check_name = args.check_name
  temp_file_name = args.temp_file_name
  temp_file_name = temp_file_name + ".cpp"

  cir_tidy_extra_args = extra_args
  cir_extra_args = []
  if '--' in extra_args:
    i = cir_tidy_extra_args.index('--')
    cir_extra_args = cir_tidy_extra_args[i + 1:]
    cir_tidy_extra_args = cir_tidy_extra_args[:i]

  # If the test does not specify a config style, force an empty one; otherwise
  # autodetection logic can discover a ".clang-tidy" file that is not related to
  # the test.
  if not any(
      [arg.startswith('-config=') for arg in cir_tidy_extra_args]):
    cir_tidy_extra_args.append('-config={}')

  with open(input_file_name, 'r', encoding='utf-8') as input_file:
    input_text = input_file.read()

  check_fixes_prefixes = []
  check_messages_prefixes = []
  check_notes_prefixes = []

  has_check_fixes = False
  has_check_messages = False
  has_check_notes = False

  check_fixes_prefix = 'CHECK-FIXES'
  check_messages_prefix = 'CHECK-MESSAGES'
  check_notes_prefix = 'CHECK-NOTES'

  has_check_fix = check_fixes_prefix in input_text
  has_check_message = check_messages_prefix in input_text
  has_check_note = check_notes_prefix in input_text

  if not has_check_fix and not has_check_message and not has_check_note:
    sys.exit('%s, %s or %s not found in the input' %
      (check_fixes_prefix, check_messages_prefix, check_notes_prefix))

  has_check_fixes = has_check_fixes or has_check_fix
  has_check_messages = has_check_messages or has_check_message
  has_check_notes = has_check_notes or has_check_note

  if has_check_fix:
      check_fixes_prefixes.append(check_fixes_prefix)
  if has_check_message:
    check_messages_prefixes.append(check_messages_prefix)
  if has_check_note:
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

  args = ['cir-tidy', temp_file_name, '--checks=-*,' + check_name] + \
      cir_tidy_extra_args + ['--'] + cir_extra_args

  arg_print_list = []
  for arg_print in cir_tidy_extra_args:
    if (arg_print.startswith("-config=")):
      conf = arg_print.replace("-config=", "-config='")
      conf += "'"
      arg_print_list.append(conf)
      continue
    arg_print_list.append(arg_print)

  cir_tidy_bin = shutil.which('cir-tidy')
  args_for_print = [cir_tidy_bin, temp_file_name, "--checks='-*," + check_name + "'"] + \
      arg_print_list + ['--'] + cir_extra_args
  print('Running: ' + " ".join(args_for_print))

  try:
    cir_tidy_output = \
        subprocess.check_output(args, stderr=subprocess.STDOUT).decode()
  except subprocess.CalledProcessError as e:
    print('cir-tidy failed:\n' + e.output.decode())
    raise

  print('------------------------ cir-tidy output -------------------------')
  print(cir_tidy_output.encode())
  print('\n------------------------------------------------------------------')

  try:
    diff_output = subprocess.check_output(
        ['diff', '-u', original_file_name, temp_file_name],
        stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError as e:
    diff_output = e.output

  print('------------------------------ Fixes -----------------------------\n' +
        diff_output.decode(errors='ignore') +
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
    write_file(messages_file, cir_tidy_output)
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
    write_file(notes_file, cir_tidy_output)
    try:
      subprocess.check_output(
          ['FileCheck', '-input-file=' + notes_file, input_file_name,
           '-check-prefixes=' + ','.join(check_notes_prefixes),
           '-implicit-check-not={{error}}:'],
          stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
      print('FileCheck failed:\n' + e.output.decode())
      raise


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file_name')
  parser.add_argument('check_name')
  parser.add_argument('temp_file_name')

  args, extra_args = parser.parse_known_args()
  run_test_once(args, extra_args)


if __name__ == '__main__':
  main()

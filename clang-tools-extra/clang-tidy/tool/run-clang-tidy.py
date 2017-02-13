#!/usr/bin/env python
#
#===- run-clang-tidy.py - Parallel clang-tidy runner ---------*- python -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
# FIXME: Integrate with clang-tidy-diff.py

"""
Parallel clang-tidy runner
==========================

Runs clang-tidy over all files in a compilation database. Requires clang-tidy
and clang-apply-replacements in $PATH.

Example invocations.
- Run clang-tidy on all files in the current working directory with a default
  set of checks and show warnings in the cpp files and all project headers.
    run-clang-tidy.py $PWD

- Fix all header guards.
    run-clang-tidy.py -fix -checks=-*,llvm-header-guard

- Fix all header guards included from clang-tidy and header guards
  for clang-tidy headers.
    run-clang-tidy.py -fix -checks=-*,llvm-header-guard extra/clang-tidy \
                      -header-filter=extra/clang-tidy

Compilation database setup:
http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
"""

import argparse
import json
import multiprocessing
import os
import Queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading


def find_compilation_database(path):
  """Adjusts the directory until a compilation database is found."""
  result = './'
  while not os.path.isfile(os.path.join(result, path)):
    if os.path.realpath(result) == '/':
      print 'Error: could not find compilation database.'
      sys.exit(1)
    result += '../'
  return os.path.realpath(result)


def get_tidy_invocation(f, clang_tidy_binary, checks, tmpdir, build_path,
                        header_filter, extra_arg, extra_arg_before, quiet):
  """Gets a command line for clang-tidy."""
  start = [clang_tidy_binary]
  if header_filter is not None:
    start.append('-header-filter=' + header_filter)
  else:
    # Show warnings in all in-project headers by default.
    start.append('-header-filter=^' + build_path + '/.*')
  if checks:
    start.append('-checks=' + checks)
  if tmpdir is not None:
    start.append('-export-fixes')
    # Get a temporary file. We immediately close the handle so clang-tidy can
    # overwrite it.
    (handle, name) = tempfile.mkstemp(suffix='.yaml', dir=tmpdir)
    os.close(handle)
    start.append(name)
  for arg in extra_arg:
      start.append('-extra-arg=%s' % arg)
  for arg in extra_arg_before:
      start.append('-extra-arg-before=%s' % arg)
  start.append('-p=' + build_path)
  if quiet:
      start.append('-quiet')
  start.append(f)
  return start


def apply_fixes(args, tmpdir):
  """Calls clang-apply-fixes on a given directory. Deletes the dir when done."""
  invocation = [args.clang_apply_replacements_binary]
  if args.format:
    invocation.append('-format')
  invocation.append(tmpdir)
  subprocess.call(invocation)
  shutil.rmtree(tmpdir)


def run_tidy(args, tmpdir, build_path, queue):
  """Takes filenames out of queue and runs clang-tidy on them."""
  while True:
    name = queue.get()
    invocation = get_tidy_invocation(name, args.clang_tidy_binary, args.checks,
                                     tmpdir, build_path, args.header_filter,
                                     args.extra_arg, args.extra_arg_before,
                                     args.quiet)
    sys.stdout.write(' '.join(invocation) + '\n')
    subprocess.call(invocation)
    queue.task_done()


def main():
  parser = argparse.ArgumentParser(description='Runs clang-tidy over all files '
                                   'in a compilation database. Requires '
                                   'clang-tidy and clang-apply-replacements in '
                                   '$PATH.')
  parser.add_argument('-clang-tidy-binary', metavar='PATH',
                      default='clang-tidy',
                      help='path to clang-tidy binary')
  parser.add_argument('-clang-apply-replacements-binary', metavar='PATH',
                      default='clang-apply-replacements',
                      help='path to clang-apply-replacements binary')
  parser.add_argument('-checks', default=None,
                      help='checks filter, when not specified, use clang-tidy '
                      'default')
  parser.add_argument('-header-filter', default=None,
                      help='regular expression matching the names of the '
                      'headers to output diagnostics from. Diagnostics from '
                      'the main file of each translation unit are always '
                      'displayed.')
  parser.add_argument('-j', type=int, default=0,
                      help='number of tidy instances to be run in parallel.')
  parser.add_argument('files', nargs='*', default=['.*'],
                      help='files to be processed (regex on path)')
  parser.add_argument('-fix', action='store_true', help='apply fix-its')
  parser.add_argument('-format', action='store_true', help='Reformat code '
                      'after applying fixes')
  parser.add_argument('-p', dest='build_path',
                      help='Path used to read a compile command database.')
  parser.add_argument('-extra-arg', dest='extra_arg',
                      action='append', default=[],
                      help='Additional argument to append to the compiler '
                      'command line.')
  parser.add_argument('-extra-arg-before', dest='extra_arg_before',
                      action='append', default=[],
                      help='Additional argument to prepend to the compiler '
                      'command line.')
  parser.add_argument('-quiet', action='store_true',
                      help='Run clang-tidy in quiet mode')
  args = parser.parse_args()

  db_path = 'compile_commands.json'

  if args.build_path is not None:
    build_path = args.build_path
  else:
    # Find our database
    build_path = find_compilation_database(db_path)

  try:
    invocation = [args.clang_tidy_binary, '-list-checks']
    invocation.append('-p=' + build_path)
    if args.checks:
      invocation.append('-checks=' + args.checks)
    invocation.append('-')
    print subprocess.check_output(invocation)
  except:
    print >>sys.stderr, "Unable to run clang-tidy."
    sys.exit(1)

  # Load the database and extract all files.
  database = json.load(open(os.path.join(build_path, db_path)))
  files = [entry['file'] for entry in database]

  max_task = args.j
  if max_task == 0:
    max_task = multiprocessing.cpu_count()

  tmpdir = None
  if args.fix:
    tmpdir = tempfile.mkdtemp()

  # Build up a big regexy filter from all command line arguments.
  file_name_re = re.compile('(' + ')|('.join(args.files) + ')')

  try:
    # Spin up a bunch of tidy-launching threads.
    queue = Queue.Queue(max_task)
    for _ in range(max_task):
      t = threading.Thread(target=run_tidy,
                           args=(args, tmpdir, build_path, queue))
      t.daemon = True
      t.start()

    # Fill the queue with files.
    for name in files:
      if file_name_re.search(name):
        queue.put(name)

    # Wait for all threads to be done.
    queue.join()

  except KeyboardInterrupt:
    # This is a sad hack. Unfortunately subprocess goes
    # bonkers with ctrl-c and we start forking merrily.
    print '\nCtrl-C detected, goodbye.'
    if args.fix:
      shutil.rmtree(tmpdir)
    os.kill(0, 9)

  if args.fix:
    print 'Applying fixes ...'
    apply_fixes(args, tmpdir)

if __name__ == '__main__':
  main()

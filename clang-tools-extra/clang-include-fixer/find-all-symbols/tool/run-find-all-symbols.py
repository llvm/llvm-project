#!/usr/bin/env python
#
#=- run-find-all-symbols.py - Parallel find-all-symbols runner -*- python  -*-=#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

"""
Parallel find-all-symbols runner
================================

Runs find-all-symbols over all files in a compilation database.

Example invocations.
- Run find-all-symbols on all files in the current working directory.
    run-find-all-symbols.py <source-file>

Compilation database setup:
http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
"""

import argparse
import json
import multiprocessing
import os
import Queue
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


def MergeSymbols(directory, args):
  """Merge all symbol files (yaml) in a given directaory into a single file."""
  invocation = [args.binary, '-merge-dir='+directory, args.saving_path]
  subprocess.call(invocation)
  print 'Merge is finished. Saving results in ' + args.saving_path


def run_find_all_symbols(args, tmpdir, build_path, queue):
  """Takes filenames out of queue and runs find-all-symbols on them."""
  while True:
    name = queue.get()
    invocation = [args.binary, name, '-output-dir='+tmpdir, '-p='+build_path]
    sys.stdout.write(' '.join(invocation) + '\n')
    subprocess.call(invocation)
    queue.task_done()


def main():
  parser = argparse.ArgumentParser(description='Runs find-all-symbols over all'
                                   'files in a compilation database.')
  parser.add_argument('-binary', metavar='PATH',
                      default='./bin/find-all-symbols',
                      help='path to find-all-symbols binary')
  parser.add_argument('-j', type=int, default=0,
                      help='number of instances to be run in parallel.')
  parser.add_argument('-p', dest='build_path',
                      help='path used to read a compilation database.')
  parser.add_argument('-saving-path', default='./find_all_symbols_db.yaml',
                      help='result saving path')
  args = parser.parse_args()

  db_path = 'compile_commands.json'

  if args.build_path is not None:
    build_path = args.build_path
  else:
    build_path = find_compilation_database(db_path)

  tmpdir = tempfile.mkdtemp()

  # Load the database and extract all files.
  database = json.load(open(os.path.join(build_path, db_path)))
  files = [entry['file'] for entry in database]

  max_task = args.j
  if max_task == 0:
    max_task = multiprocessing.cpu_count()

  try:
    # Spin up a bunch of tidy-launching threads.
    queue = Queue.Queue(max_task)
    for _ in range(max_task):
      t = threading.Thread(target=run_find_all_symbols,
                           args=(args, tmpdir, build_path, queue))
      t.daemon = True
      t.start()

    # Fill the queue with files.
    for name in files:
      queue.put(name)

    # Wait for all threads to be done.
    queue.join()

    MergeSymbols(tmpdir, args)


  except KeyboardInterrupt:
    # This is a sad hack. Unfortunately subprocess goes
    # bonkers with ctrl-c and we start forking merrily.
    print '\nCtrl-C detected, goodbye.'
    os.kill(0, 9)


if __name__ == '__main__':
  main()

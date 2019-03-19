#!/usr/bin/env python
"""Calls C-Reduce to create a minimal reproducer for clang crashes.

Requires C-Reduce and not (part of LLVM utils) to be installed.
"""

from argparse import ArgumentParser
import os
import re
import stat
import sys
import subprocess
import pipes
from distutils.spawn import find_executable

def create_test(build_script, llvm_not):
  """
  Create an interestingness test from the crash output.
  Return as a string.
  """
  # Get clang call from build script
  # Assumes the call is the last line of the script
  with open(build_script) as f:
    cmd = f.readlines()[-1].rstrip('\n\r')

  # Get crash output
  p = subprocess.Popen(build_script,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
  crash_output, _ = p.communicate()

  output = ['#!/bin/bash']
  output.append('%s --crash %s >& t.log || exit 1' % (pipes.quote(llvm_not),
                                                      cmd))

  # Add messages from crash output to the test
  # If there is an Assertion failure, use that; otherwise use the
  # last five stack trace functions
  assertion_re = r'Assertion `([^\']+)\' failed'
  assertion_match = re.search(assertion_re, crash_output)
  if assertion_match:
    msg = assertion_match.group(1)
    output.append('grep %s t.log || exit 1' % pipes.quote(msg))
  else:
    stacktrace_re = r'#[0-9]+\s+0[xX][0-9a-fA-F]+\s*([^(]+)\('
    matches = re.findall(stacktrace_re, crash_output)
    del matches[:-5]
    output += ['grep %s t.log || exit 1' % pipes.quote(msg) for msg in matches]

  return output

def main():
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('build_script', type=str, nargs=1,
                      help='Name of the script that generates the crash.')
  parser.add_argument('file_to_reduce', type=str, nargs=1,
                      help='Name of the file to be reduced.')
  parser.add_argument('-o', '--output', dest='output', type=str,
                      help='Name of the output file for the reduction. Optional.')
  parser.add_argument('--llvm-not', dest='llvm_not', type=str,
                      help="The path to the llvm-not executable. "
                      "Required if 'not' is not in PATH environment.");
  parser.add_argument('--creduce', dest='creduce', type=str,
                      help="The path to the C-Reduce executable. "
                      "Required if 'creduce' is not in PATH environment.");
  args = parser.parse_args()

  build_script = os.path.abspath(args.build_script[0])
  file_to_reduce = os.path.abspath(args.file_to_reduce[0])
  llvm_not = (find_executable(args.llvm_not) if args.llvm_not else
              find_executable('not'))
  creduce = (find_executable(args.creduce) if args.creduce else
             find_executable('creduce'))

  if not os.path.isfile(build_script):
    print(("ERROR: input file '%s' does not exist") % build_script)
    return 1

  if not os.path.isfile(file_to_reduce):
    print(("ERROR: input file '%s' does not exist") % file_to_reduce)
    return 1

  if not llvm_not:
    parser.print_help()
    return 1

  if not creduce:
    parser.print_help()
    return 1

  # Write interestingness test to file
  test_contents = create_test(build_script, llvm_not)
  testname, _ = os.path.splitext(file_to_reduce)
  testfile = testname + '.test.sh'
  with open(testfile, 'w') as f:
    f.write('\n'.join(test_contents))
  os.chmod(testfile, os.stat(testfile).st_mode | stat.S_IEXEC)

  # Confirm that the interestingness test passes
  try:
    with open(os.devnull, 'w') as devnull:
      subprocess.check_call(testfile, stdout=devnull)
  except subprocess.CalledProcessError:
    print("For some reason the interestingness test does not return zero")
    return 1

  # FIXME: try running clang preprocessor first

  try:
    p = subprocess.Popen([creduce, testfile, file_to_reduce])
    p.communicate()
  except KeyboardInterrupt:
    # Hack to kill C-Reduce because it jumps into its own pgid
    print('\n\nctrl-c detected, killed creduce')
    p.kill()

if __name__ == '__main__':
  sys.exit(main())

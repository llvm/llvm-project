#!/usr/bin/env python
"""Calls C-Reduce to create a minimal reproducer for clang crashes.
"""

from argparse import ArgumentParser
import os
import re
import stat
import sys
import subprocess
import pipes
import shlex
import tempfile
import shutil
from distutils.spawn import find_executable

verbose = False
llvm_bin = None
creduce_cmd = None
not_cmd = None

def check_file(fname):
  if not os.path.isfile(fname):
    sys.exit("ERROR: %s does not exist" % (fname))
  return fname

def check_cmd(cmd_name, cmd_dir, cmd_path=None):
  """
  Returns absolute path to cmd_path if it is given,
  or absolute path to cmd_dir/cmd_name.
  """
  if cmd_path:
    cmd = find_executable(cmd_path)
    if cmd:
      return cmd
    sys.exit("ERROR: executable %s not found" % (cmd_path))

  cmd = find_executable(cmd_name, path=cmd_dir)
  if cmd:
    return cmd
  sys.exit("ERROR: %s not found in %s" % (cmd_name, cmd_dir))

def quote_cmd(cmd):
  return ' '.join(arg if arg.startswith('$') else pipes.quote(arg)
                  for arg in cmd)

def get_crash_cmd(crash_script):
  with open(crash_script) as f:
    # Assume clang call is on the last line of the script
    line = f.readlines()[-1]
    cmd = shlex.split(line)

    # Overwrite the script's clang with the user's clang path
    new_clang = check_cmd('clang', llvm_bin)
    cmd[0] = pipes.quote(new_clang)
    return cmd

def has_expected_output(crash_cmd, expected_output):
  p = subprocess.Popen(crash_cmd,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
  crash_output, _ = p.communicate()
  return all(msg in crash_output for msg in expected_output)

def get_expected_output(crash_cmd):
  p = subprocess.Popen(crash_cmd,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
  crash_output, _ = p.communicate()

  # If there is an assertion failure, use that;
  # otherwise use the last five stack trace functions
  assertion_re = r'Assertion `([^\']+)\' failed'
  assertion_match = re.search(assertion_re, crash_output)
  if assertion_match:
    return [assertion_match.group(1)]
  else:
    stacktrace_re = r'#[0-9]+\s+0[xX][0-9a-fA-F]+\s*([^(]+)\('
    matches = re.findall(stacktrace_re, crash_output)
    return matches[-5:]

def write_interestingness_test(testfile, crash_cmd, expected_output,
                               file_to_reduce):
  filename = os.path.basename(file_to_reduce)
  if filename not in crash_cmd:
    sys.exit("ERROR: expected %s to be in the crash command" % filename)

  # Replace all instances of file_to_reduce with a command line variable
  output = ['#!/bin/bash',
            'if [ -z "$1" ] ; then',
            '  f=%s' % (pipes.quote(filename)),
            'else',
            '  f="$1"',
            'fi']
  cmd = ['$f' if s == filename else s for s in crash_cmd]

  output.append('%s --crash %s >& t.log || exit 1' % (pipes.quote(not_cmd),
                                                      quote_cmd(cmd)))

  for msg in expected_output:
    output.append('grep %s t.log || exit 1' % pipes.quote(msg))

  with open(testfile, 'w') as f:
    f.write('\n'.join(output))
  os.chmod(testfile, os.stat(testfile).st_mode | stat.S_IEXEC)

def check_interestingness(testfile, file_to_reduce):
  testfile = os.path.abspath(testfile)

  # Check that the test considers the original file interesting
  with open(os.devnull, 'w') as devnull:
    returncode = subprocess.call(testfile, stdout=devnull)
  if returncode:
    sys.exit("The interestingness test does not pass for the original file.")

  # Check that an empty file is not interesting
  _, empty_file = tempfile.mkstemp()
  with open(os.devnull, 'w') as devnull:
    returncode = subprocess.call([testfile, empty_file], stdout=devnull)
  os.remove(empty_file)
  if not returncode:
    sys.exit("The interestingness test passes for an empty file.")

def clang_preprocess(file_to_reduce, crash_cmd, expected_output):
  _, tmpfile = tempfile.mkstemp()
  shutil.copy(file_to_reduce, tmpfile)

  cmd = crash_cmd + ['-E', '-P']
  p = subprocess.Popen(cmd,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
  preprocessed, _ = p.communicate()

  with open(file_to_reduce, 'w') as f:
    f.write(preprocessed)

  if has_expected_output(crash_cmd, expected_output):
    if verbose:
      print("Successfuly preprocessed with %s" % (quote_cmd(cmd)))
    os.remove(tmpfile)
  else:
    if verbose:
      print("Failed to preprocess with %s" % (quote_cmd(cmd)))
    shutil.move(tmpfile, file_to_reduce)


def filter_args(args, opts_startswith=[]):
  result = [arg for arg in args if all(not arg.startswith(a) for a in
                                       opts_startswith)]
  return result

def try_remove_args(cmd, expected_output, msg=None, extra_arg=None, **kwargs):
  new_cmd = filter_args(cmd, **kwargs)
  if extra_arg and extra_arg not in new_cmd:
    new_cmd.append(extra_arg)
  if new_cmd != cmd and has_expected_output(new_cmd, expected_output):
    if msg and verbose:
      print(msg)
    return new_cmd
  return cmd

def simplify_crash_cmd(crash_cmd, expected_output):
  new_cmd = try_remove_args(crash_cmd, expected_output,
                            msg="Removed debug info options",
                            opts_startswith=["-gcodeview",
                                             "-dwarf-column-info",
                                             "-debug-info-kind=",
                                             "-debugger-tuning=",
                                             "-gdwarf"])
  new_cmd = try_remove_args(new_cmd, expected_output,
                            msg="Replaced -W options with -w",
                            extra_arg='-w',
                            opts_startswith=["-W"])
  new_cmd = try_remove_args(new_cmd, expected_output,
                            msg="Replaced optimization level with -O0",
                            extra_arg="-O0",
                            opts_startswith=["-O"])
  return new_cmd

def main():
  global verbose
  global llvm_bin
  global creduce_cmd
  global not_cmd

  parser = ArgumentParser(description=__doc__)
  parser.add_argument('crash_script', type=str, nargs=1,
                      help="Name of the script that generates the crash.")
  parser.add_argument('file_to_reduce', type=str, nargs=1,
                      help="Name of the file to be reduced.")
  parser.add_argument('--llvm-bin', dest='llvm_bin', type=str,
                      required=True, help="Path to the LLVM bin directory.")
  parser.add_argument('--llvm-not', dest='llvm_not', type=str,
                      help="The path to the `not` executable. "
                      "By default uses the llvm-bin directory.")
  parser.add_argument('--creduce', dest='creduce', type=str,
                      help="The path to the `creduce` executable. "
                      "Required if `creduce` is not in PATH environment.")
  parser.add_argument('-v', '--verbose', action='store_true')
  args = parser.parse_args()

  verbose = args.verbose
  llvm_bin = os.path.abspath(args.llvm_bin)
  creduce_cmd = check_cmd('creduce', None, args.creduce)
  not_cmd = check_cmd('not', llvm_bin, args.llvm_not)
  crash_script = check_file(args.crash_script[0])
  file_to_reduce = check_file(args.file_to_reduce[0])

  print("\nParsing the crash script and getting expected output...")
  crash_cmd = get_crash_cmd(crash_script)

  expected_output = get_expected_output(crash_cmd)
  if len(expected_output) < 1:
    sys.exit("ERROR: no crash was found")

  print("\nSimplifying the crash command...")
  crash_cmd = simplify_crash_cmd(crash_cmd, expected_output)

  print("\nWriting interestingness test to file...")
  testfile = os.path.splitext(file_to_reduce)[0] + '.test.sh'
  write_interestingness_test(testfile, crash_cmd, expected_output,
                             file_to_reduce)
  check_interestingness(testfile, file_to_reduce)

  print("\nPreprocessing the file to reduce...")
  clang_preprocess(file_to_reduce, crash_cmd, expected_output)

  print("\nRunning C-Reduce...")
  try:
    p = subprocess.Popen([creduce_cmd, testfile, file_to_reduce])
    p.communicate()
  except KeyboardInterrupt:
    # Hack to kill C-Reduce because it jumps into its own pgid
    print('\n\nctrl-c detected, killed creduce')
    p.kill()

  # FIXME: reduce the clang crash command

if __name__ == '__main__':
  main()

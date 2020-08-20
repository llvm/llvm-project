#!/usr/bin/env python
#
# litlint
#
# Ensure RUN commands in lit tests are free of common errors.
#
# If any errors are detected, litlint returns a nonzero exit code.
#

import optparse
import re
import sys

# Compile regex once for all files
runRegex = re.compile(r'(?<!-o)(?<!%run) %t\s')

def LintLine(s):
  """ Validate a line

  Args:
    s: str, the line to validate

  Returns:
    Returns an error message and a 1-based column number if an error was
    detected, otherwise (None, None).
  """

  # Check that RUN command can be executed with an emulator
  m = runRegex.search(s)
  if m:
    start, end = m.span()
    return ('missing %run before %t', start + 2)

  # No errors
  return (None, None)


def LintFile(p):
  """ Check that each RUN command can be executed with an emulator

  Args:
    p: str, valid path to a file

  Returns:
    The number of errors detected.
  """
  errs = 0
  with open(p, 'r', encoding='utf-8') as f:
    for i, s in enumerate(f.readlines(), start=1):
      msg, col = LintLine(s)
      if msg != None:
        errs += 1
        errorMsg = 'litlint: {}:{}:{}: error: {}.\n{}{}\n'
        arrow = (col-1) * ' ' + '^'
        sys.stderr.write(errorMsg.format(p, i, col, msg, s, arrow))
  return errs


if __name__ == "__main__":
  # Parse args
  parser = optparse.OptionParser()
  parser.add_option('--filter')  # ignored
  (options, filenames) = parser.parse_args()

  # Lint each file
  errs = 0
  for p in filenames:
    errs += LintFile(p)

  # If errors, return nonzero
  if errs > 0:
    sys.exit(1)

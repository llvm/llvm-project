# This file is a minimal clang-include-fixer vim-integration. To install:
# - Change 'binary' if clang-include-fixer is not on the path (see below).
# - Add to your .vimrc:
#
#   map ,cf :pyf path/to/llvm/source/tools/clang/tools/extra/include-fixer/tool/clang-include-fixer.py<cr>
#
# This enables clang-include-fixer for NORMAL and VISUAL mode. Change ",cf" to
# another binding if you need clang-include-fixer on a different key.
#
# To set up clang-include-fixer, see http://clang.llvm.org/extra/include-fixer.html
#
# With this integration you can press the bound key and clang-include-fixer will
# be run on the current buffer.
#
# It operates on the current, potentially unsaved buffer and does not create
# or save any files. To revert a fix, just undo.

import argparse
import difflib
import subprocess
import vim
import json

# set g:clang_include_fixer_path to the path to clang-include-fixer if it is not
# on the path.
# Change this to the full path if clang-include-fixer is not on the path.
binary = 'clang-include-fixer'
if vim.eval('exists("g:clang_include_fixer_path")') == "1":
  binary = vim.eval('g:clang_include_fixer_path')

maximum_suggested_headers=3
if vim.eval('exists("g:clang_include_fixer_maximum_suggested_headers")') == "1":
  maximum_suggested_headers = max(
      1,
      vim.eval('g:clang_include_fixer_maximum_suggested_headers'))


def GetUserSelection(message, headers, maximum_suggested_headers):
  eval_message = message + '\n'
  for idx, header in enumerate(headers[0:maximum_suggested_headers]):
    eval_message += "({0}). {1}\n".format(idx+1, header)
  eval_message += "Enter (q) to quit;"
  if maximum_suggested_headers < len(headers):
    eval_message += " (a) to show all candidates.";
  eval_message += "\nSelect (default 1): "
  res = vim.eval("input('{0}')".format(eval_message))
  if res == '':
    # choose the top ranked header by default
    idx = 1
  elif res == 'q':
    raise Exception('   Insertion cancelled...')
  elif res == 'a' and maximum_suggested_headers < len(headers):
    return GetUserSelection(message, headers, len(headers))
  else:
    try:
      idx = int(res)
      if idx <= 0 or idx > len(headers):
        raise Exception()
    except Exception:
        raise Exception('   ERROR: Invalid option "{0}"...Abort!'.format(res))
  return headers[idx-1]

def execute(command, text):
  p = subprocess.Popen(command,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.PIPE)
  return p.communicate(input=text)


def InsertHeaderToVimBuffer(header, text):
  command = [binary, "-stdin", "-insert-header="+json.dumps(header),
             vim.current.buffer.name]
  stdout, stderr = execute(command, text)
  if stdout:
    lines = stdout.splitlines()
    sequence = difflib.SequenceMatcher(None, vim.current.buffer, lines)
    for op in reversed(sequence.get_opcodes()):
      if op[0] is not 'equal':
        vim.current.buffer[op[1]:op[2]] = lines[op[3]:op[4]]


def main():
  parser = argparse.ArgumentParser(
      description='Vim integration for clang-include-fixer')
  parser.add_argument('-db', default='yaml',
                      help='clang-include-fixer input format.')
  parser.add_argument('-input', default='',
                      help='String to initialize the database.')
  args = parser.parse_args()

  # Get the current text.
  buf = vim.current.buffer
  text = '\n'.join(buf)

  # Run command to get all headers.
  command = [binary, "-stdin", "-output-headers", "-db="+args.db,
             "-input="+args.input, vim.current.buffer.name]
  stdout, stderr = execute(command, text)
  if stderr:
    print >> sys.stderr, "Error while running clang-include-fixer: " + stderr
    return

  include_fixer_context = json.loads(stdout)
  symbol = include_fixer_context["SymbolIdentifier"]
  headers = include_fixer_context["Headers"]

  if not symbol:
    print "The file is fine, no need to add a header.\n"
    return

  if not headers:
    print "Couldn't find a header for {0}.\n".format(symbol)
    return

  # The first line is the symbol name.
  # If there is only one suggested header, insert it directly.
  if len(headers) == 1 or maximum_suggested_headers == 1:
    InsertHeaderToVimBuffer({"SymbolIdentifier": symbol,
                             "Headers":[headers[0]]}, text)
    print "Added #include {0} for {1}.\n".format(headers[0], symbol)
    return

  try:
    selected = GetUserSelection("choose a header file for {0}.".format(symbol),
                                headers, maximum_suggested_headers)
    # Insert a selected header.
    InsertHeaderToVimBuffer({"SymbolIdentifier": symbol,
                             "Headers":[selected]}, text)
    print "Added #include {0} for {1}.\n".format(selected, symbol)
  except Exception as error:
    print >> sys.stderr, error.message
  return


if __name__ == '__main__':
  main()

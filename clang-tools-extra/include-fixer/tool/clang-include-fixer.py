# This file is a minimal clang-include-fixer vim-integration. To install:
# - Change 'binary' if clang-include-fixer is not on the path (see below).
# - Add to your .vimrc:
#
#   noremap <leader>cf :pyf path/to/llvm/source/tools/clang/tools/extra/include-fixer/tool/clang-include-fixer.py<cr>
#
# This enables clang-include-fixer for NORMAL and VISUAL mode. Change "<leader>cf"
# to another binding if you need clang-include-fixer on a different key.
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
import json
import re
import subprocess
import vim

# set g:clang_include_fixer_path to the path to clang-include-fixer if it is not
# on the path.
# Change this to the full path if clang-include-fixer is not on the path.
binary = 'clang-include-fixer'
if vim.eval('exists("g:clang_include_fixer_path")') == "1":
  binary = vim.eval('g:clang_include_fixer_path')

maximum_suggested_headers = 3
if vim.eval('exists("g:clang_include_fixer_maximum_suggested_headers")') == "1":
  maximum_suggested_headers = max(
      1,
      vim.eval('g:clang_include_fixer_maximum_suggested_headers'))

increment_num = 5
if vim.eval('exists("g:clang_include_fixer_increment_num")') == "1":
  increment_num = max(
      1,
      vim.eval('g:clang_include_fixer_increment_num'))

jump_to_include = False
if vim.eval('exists("g:clang_include_fixer_jump_to_include")') == "1":
  jump_to_include = vim.eval('g:clang_include_fixer_jump_to_include') != "0"

query_mode = False
if vim.eval('exists("g:clang_include_fixer_query_mode")') == "1":
  query_mode = vim.eval('g:clang_include_fixer_query_mode') != "0"


def GetUserSelection(message, headers, maximum_suggested_headers):
  eval_message = message + '\n'
  for idx, header in enumerate(headers[0:maximum_suggested_headers]):
    eval_message += "({0}). {1}\n".format(idx + 1, header)
  eval_message += "Enter (q) to quit;"
  if maximum_suggested_headers < len(headers):
    eval_message += " (m) to show {0} more candidates.".format(
        min(increment_num, len(headers) - maximum_suggested_headers))

  eval_message += "\nSelect (default 1): "
  res = vim.eval("input('{0}')".format(eval_message))
  if res == '':
    # choose the top ranked header by default
    idx = 1
  elif res == 'q':
    raise Exception('   Insertion cancelled...')
  elif res == 'm':
    return GetUserSelection(message,
                            headers, maximum_suggested_headers + increment_num)
  else:
    try:
      idx = int(res)
      if idx <= 0 or idx > len(headers):
        raise Exception()
    except Exception:
      # Show a new prompt on invalid option instead of aborting so that users
      # don't need to wait for another include-fixer run.
      print >> sys.stderr, "Invalid option:", res
      return GetUserSelection(message, headers, maximum_suggested_headers)
  return headers[idx - 1]


def execute(command, text):
  p = subprocess.Popen(command,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.PIPE)
  return p.communicate(input=text)


def InsertHeaderToVimBuffer(header, text):
  command = [binary, "-stdin", "-insert-header=" + json.dumps(header),
             vim.current.buffer.name]
  stdout, stderr = execute(command, text)
  if stderr:
    raise Exception(stderr)
  if stdout:
    lines = stdout.splitlines()
    sequence = difflib.SequenceMatcher(None, vim.current.buffer, lines)
    line_num = None
    for op in reversed(sequence.get_opcodes()):
      if op[0] != 'equal':
        vim.current.buffer[op[1]:op[2]] = lines[op[3]:op[4]]
      if op[0] == 'insert':
        # line_num in vim is 1-based.
        line_num = op[1] + 1

    if jump_to_include and line_num:
      vim.current.window.cursor = (line_num, 0)


# The vim internal implementation (expand("cword"/"cWORD")) doesn't support
# our use case very well, we re-implement our own one.
def get_symbol_under_cursor():
  line = vim.eval("line(\".\")")
  # column number in vim is 1-based.
  col = int(vim.eval("col(\".\")")) - 1
  line_text = vim.eval("getline({0})".format(line))
  if len(line_text) == 0: return ""
  symbol_pos_begin = col
  p = re.compile('[a-zA-Z0-9:_]')
  while symbol_pos_begin >= 0 and p.match(line_text[symbol_pos_begin]):
    symbol_pos_begin -= 1

  symbol_pos_end = col
  while symbol_pos_end < len(line_text) and p.match(line_text[symbol_pos_end]):
    symbol_pos_end += 1
  return line_text[symbol_pos_begin+1:symbol_pos_end]


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

  if query_mode:
    symbol = get_symbol_under_cursor()
    if len(symbol) == 0:
      print "Skip querying empty symbol."
      return
    command = [binary, "-stdin", "-query-symbol="+get_symbol_under_cursor(),
               "-db=" + args.db, "-input=" + args.input,
               vim.current.buffer.name]
  else:
    # Run command to get all headers.
    command = [binary, "-stdin", "-output-headers", "-db=" + args.db,
               "-input=" + args.input, vim.current.buffer.name]
  stdout, stderr = execute(command, text)
  if stderr:
    print >> sys.stderr, "Error while running clang-include-fixer: " + stderr
    return

  include_fixer_context = json.loads(stdout)
  query_symbol_infos = include_fixer_context["QuerySymbolInfos"]
  if not query_symbol_infos:
    print "The file is fine, no need to add a header."
    return
  symbol = query_symbol_infos[0]["RawIdentifier"]
  # The header_infos is already sorted by include-fixer.
  header_infos = include_fixer_context["HeaderInfos"]
  # Deduplicate headers while keeping the order, so that the same header would
  # not be suggested twice.
  unique_headers = []
  seen = set()
  for header_info in header_infos:
    header = header_info["Header"]
    if header not in seen:
      seen.add(header)
      unique_headers.append(header)

  if not unique_headers:
    print "Couldn't find a header for {0}.".format(symbol)
    return

  try:
    selected = unique_headers[0]
    inserted_header_infos = header_infos
    if len(unique_headers) > 1:
      selected = GetUserSelection(
          "choose a header file for {0}.".format(symbol),
          unique_headers, maximum_suggested_headers)
      inserted_header_infos = [
        header for header in header_infos if header["Header"] == selected]
    include_fixer_context["HeaderInfos"] = inserted_header_infos

    InsertHeaderToVimBuffer(include_fixer_context, text)
    print "Added #include {0} for {1}.".format(selected, symbol)
  except Exception as error:
    print >> sys.stderr, error.message
  return


if __name__ == '__main__':
  main()

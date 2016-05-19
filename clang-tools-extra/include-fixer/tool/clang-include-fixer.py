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
import subprocess
import sys
import vim

# set g:clang_include_fixer_path to the path to clang-include-fixer if it is not
# on the path.
# Change this to the full path if clang-include-fixer is not on the path.
binary = 'clang-include-fixer'
if vim.eval('exists("g:clang_include_fixer_path")') == "1":
  binary = vim.eval('g:clang_include_fixer_path')

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

  # Call clang-include-fixer.
  command = [binary, "-stdin", "-db="+args.db, "-input="+args.input,
             vim.current.buffer.name]
  p = subprocess.Popen(command,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.PIPE)
  stdout, stderr = p.communicate(input=text)

  # If successful, replace buffer contents.
  if stderr:
    print stderr

  if stdout:
    lines = stdout.splitlines()
    for line in lines:
      line_num, text = line.split(",")
      # clang-include-fixer provides 1-based line number
      line_num = int(line_num) - 1
      print 'Inserting "{0}" at line {1}'.format(text, line_num)
      vim.current.buffer[line_num:line_num] = [text]

if __name__ == '__main__':
  main()

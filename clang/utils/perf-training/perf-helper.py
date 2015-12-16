#===- perf-helper.py - Clang Python Bindings -----------------*- python -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

import sys
import os
import subprocess

def findProfrawFiles(path):
  profraw_files = []
  for root, dirs, files in os.walk(path): 
    for filename in files:
      if filename.endswith(".profraw"):
        profraw_files.append(os.path.join(root, filename))
  return profraw_files

def clean(args):
  if len(args) != 1:
    print 'Usage: %s clean <path>\n\tRemoves all *.profraw files from <path>.' % __file__
    return 1
  for profraw in findProfrawFiles(args[0]):
    os.remove(profraw)
  return 0

def merge(args):
  if len(args) != 3:
    print 'Usage: %s clean <llvm-profdata> <output> <path>\n\tMerges all profraw files from path into output.' % __file__
    return 1
  cmd = [args[0], 'merge', '-o', args[1]]
  cmd.extend(findProfrawFiles(args[2]))
  subprocess.check_call(cmd)
  return 0

commands = {'clean' : clean, 'merge' : merge}

def main():
  f = commands[sys.argv[1]]
  sys.exit(f(sys.argv[2:]))

if __name__ == '__main__':
  main()

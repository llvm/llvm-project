#!/usr/bin/env python
#===- lib/sanitizer_common/scripts/gen_dynamic_list.py ---------------------===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
#
# Generates the list of functions that should be exported from sanitizer
# runtimes. The output format is recognized by --dynamic-list linker option.
# Usage:
#   gen_dynamic_list.py libclang_rt.*san*.a [ files ... ]
#
#===------------------------------------------------------------------------===#
import argparse
import os
import re
import subprocess
import sys
import platform

new_delete = set([
                  '_Znam', '_ZnamRKSt9nothrow_t',    # operator new[](unsigned long)
                  '_Znwm', '_ZnwmRKSt9nothrow_t',    # operator new(unsigned long)
                  '_Znaj', '_ZnajRKSt9nothrow_t',    # operator new[](unsigned int)
                  '_Znwj', '_ZnwjRKSt9nothrow_t',    # operator new(unsigned int)
                  '_ZdaPv', '_ZdaPvRKSt9nothrow_t',  # operator delete[](void *)
                  '_ZdlPv', '_ZdlPvRKSt9nothrow_t',  # operator delete(void *)
                  '_ZdaPvm',                         # operator delete[](void*, unsigned long)
                  '_ZdlPvm',                         # operator delete(void*, unsigned long)
                  '_ZdaPvj',                         # operator delete[](void*, unsigned int)
                  '_ZdlPvj',                         # operator delete(void*, unsigned int)
                  ])

versioned_functions = set(['memcpy', 'pthread_attr_getaffinity_np',
                           'pthread_cond_broadcast',
                           'pthread_cond_destroy', 'pthread_cond_init',
                           'pthread_cond_signal', 'pthread_cond_timedwait',
                           'pthread_cond_wait', 'realpath',
                           'sched_getaffinity'])

def get_global_functions(library):
  functions = []
  nm = os.environ.get('NM', 'nm')
  nm_proc = subprocess.Popen([nm, library], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
  nm_out = nm_proc.communicate()[0].decode().split('\n')
  if nm_proc.returncode != 0:
    raise subprocess.CalledProcessError(nm_proc.returncode, nm)
  func_symbols = ['T', 'W']
  # On PowerPC, nm prints function descriptors from .data section.
  if platform.uname()[4] in ["powerpc", "ppc64"]:
    func_symbols += ['D']
  for line in nm_out:
    cols = line.split(' ')
    if len(cols) == 3 and cols[1] in func_symbols :
      functions.append(cols[2])
  return functions

def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--version-list', action='store_true')
  parser.add_argument('--extra', default=[], action='append')
  parser.add_argument('libraries', default=[], nargs='+')
  args = parser.parse_args()

  result = []

  all_functions = []
  for library in args.libraries:
    all_functions.extend(get_global_functions(library))
  function_set = set(all_functions)
  for func in all_functions:
    # Export new/delete operators.
    if func in new_delete:
      result.append(func)
      continue
    # Export interceptors.
    match = re.match('__interceptor_(.*)', func)
    if match:
      result.append(func)
      # We have to avoid exporting the interceptors for versioned library
      # functions due to gold internal error.
      orig_name = match.group(1)
      if orig_name in function_set and (args.version_list or orig_name not in versioned_functions):
        result.append(orig_name)
      continue
    # Export sanitizer interface functions.
    if re.match('__sanitizer_(.*)', func):
      result.append(func)

  # Additional exported functions from files.
  for fname in args.extra:
    f = open(fname, 'r')
    for line in f:
      result.append(line.rstrip())
  # Print the resulting list in the format recognized by ld.
  print('{')
  if args.version_list:
    print('global:')
  result.sort()
  for f in result:
    print(u'  %s;' % f)
  if args.version_list:
    print('local:')
    print('  *;')
  print('};')

if __name__ == '__main__':
  main(sys.argv)

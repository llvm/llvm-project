#!/usr/bin/env python
#===- lib/dfsan/scripts/build-libc-list.py ---------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#
# The purpose of this script is to identify every function symbol in a set of
# libraries (in this case, libc and libgcc) so that they can be marked as
# uninstrumented, thus allowing the instrumentation pass to treat calls to those
# functions correctly.

import os
import subprocess
import sys
from optparse import OptionParser

def defined_function_list(object):
  functions = []
  readelf_proc = subprocess.Popen(['readelf', '-s', '-W', object],
                                  stdout=subprocess.PIPE)
  readelf = readelf_proc.communicate()[0].split('\n')
  if readelf_proc.returncode != 0:
    raise subprocess.CalledProcessError(readelf_proc.returncode, 'readelf')
  for line in readelf:
    if (line[31:35] == 'FUNC' or line[31:36] == 'IFUNC') and \
       line[39:44] != 'LOCAL' and \
       line[55:58] != 'UND':
      function_name = line[59:].split('@')[0]
      functions.append(function_name)
  return functions

p = OptionParser()

p.add_option('--libc-dso-path', metavar='PATH',
             help='path to libc DSO directory',
             default='/lib/x86_64-linux-gnu')
p.add_option('--libc-archive-path', metavar='PATH',
             help='path to libc archive directory',
             default='/usr/lib/x86_64-linux-gnu')

p.add_option('--libgcc-dso-path', metavar='PATH',
             help='path to libgcc DSO directory',
             default='/lib/x86_64-linux-gnu')
p.add_option('--libgcc-archive-path', metavar='PATH',
             help='path to libgcc archive directory',
             default='/usr/lib/gcc/x86_64-linux-gnu/4.6')

p.add_option('--with-libstdcxx', action='store_true',
             dest='with_libstdcxx',
             help='include libstdc++ in the list (inadvisable)')
p.add_option('--libstdcxx-dso-path', metavar='PATH',
             help='path to libstdc++ DSO directory',
             default='/usr/lib/x86_64-linux-gnu')

p.add_option('--only-explicit-files', action='store_true',
             dest='only_explicit_files', default=False,
             help='Only process --lib-file, not the default libc libraries.')
p.add_option('--lib-file', action='append', metavar='PATH',
             help='Specific library files to add.',
             default=[])

p.add_option('--error-missing-lib', action='store_true',
             help='Make this script exit with an error code if any library is missing.',
             dest='error_missing_lib', default=False)

(options, args) = p.parse_args()

def build_libs_list():
    libs = [os.path.join(options.libc_dso_path, name) for name in
            ['ld-linux-x86-64.so.2',
             'libanl.so.1',
             'libBrokenLocale.so.1',
             'libcidn.so.1',
             'libcrypt.so.1',
             'libc.so.6',
             'libdl.so.2',
             'libm.so.6',
             'libnsl.so.1',
             'libpthread.so.0',
             'libresolv.so.2',
             'librt.so.1',
             'libthread_db.so.1',
             'libutil.so.1']]
    libs += [os.path.join(options.libc_archive_path, name) for name in
             ['libc_nonshared.a',
              'libpthread_nonshared.a']]

    libs.append(os.path.join(options.libgcc_dso_path, 'libgcc_s.so.1'))
    libs.append(os.path.join(options.libgcc_archive_path, 'libgcc.a'))

    if options.with_libstdcxx:
      libs.append(os.path.join(options.libstdcxx_dso_path, 'libstdc++.so.6'))

    return libs

libs = []
if options.only_explicit_files:
    libs = options.lib_file
    if not libs:
        print >> sys.stderr, 'No libraries provided.'
        exit(1)
else:
    libs = build_libs_list()
    libs.extend(options.lib_file)

missing_lib = False
functions = []
for l in libs:
  if os.path.exists(l):
    functions += defined_function_list(l)
  else:
    missing_lib = True
    print >> sys.stderr, 'warning: library %s not found' % l

if options.error_missing_lib and missing_lib:
    print >> sys.stderr, 'Exiting with failure code due to missing library.'
    exit(1)

functions = list(set(functions))
functions.sort()

for f in functions:
  print 'fun:%s=uninstrumented' % f

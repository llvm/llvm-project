#!/usr/bin/python3
# Shared code for glibc conformance tests.
# Copyright (C) 2018-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
#
# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

import os.path
import re
import subprocess
import tempfile


# Compiler options for each standard.
CFLAGS = {'ISO': '-ansi',
          'ISO99': '-std=c99',
          'ISO11': '-std=c11',
          'POSIX': '-D_POSIX_C_SOURCE=199506L -ansi',
          'XPG4': '-ansi -D_XOPEN_SOURCE',
          'XPG42': '-ansi -D_XOPEN_SOURCE -D_XOPEN_SOURCE_EXTENDED',
          'UNIX98': '-ansi -D_XOPEN_SOURCE=500',
          'XOPEN2K': '-std=c99 -D_XOPEN_SOURCE=600',
          'XOPEN2K8': '-std=c99 -D_XOPEN_SOURCE=700',
          'POSIX2008': '-std=c99 -D_POSIX_C_SOURCE=200809L'}

# ISO C90 keywords.
KEYWORDS_C90 = {'auto', 'break', 'case', 'char', 'const', 'continue',
                'default', 'do', 'double', 'else', 'enum', 'extern', 'float',
                'for', 'goto', 'if', 'int', 'long', 'register', 'return',
                'short', 'signed', 'sizeof', 'static', 'struct', 'switch',
                'typedef', 'union', 'unsigned', 'void', 'volatile', 'while'}

# ISO C99 keywords.
KEYWORDS_C99 = KEYWORDS_C90 | {'inline', 'restrict'}

# Keywords for each standard.
KEYWORDS = {'ISO': KEYWORDS_C90,
            'ISO99': KEYWORDS_C99,
            'ISO11': KEYWORDS_C99,
            'POSIX': KEYWORDS_C90,
            'XPG4': KEYWORDS_C90,
            'XPG42': KEYWORDS_C90,
            'UNIX98': KEYWORDS_C90,
            'XOPEN2K': KEYWORDS_C99,
            'XOPEN2K8': KEYWORDS_C99,
            'POSIX2008': KEYWORDS_C99}


def list_exported_functions(cc, standard, header):
    """Return the set of functions exported by a header, empty if an
    include of the header does not compile.

    """
    cc_all = '%s -D_ISOMAC %s' % (cc, CFLAGS[standard])
    with tempfile.TemporaryDirectory() as temp_dir:
        c_file_name = os.path.join(temp_dir, 'test.c')
        aux_file_name = os.path.join(temp_dir, 'test.c.aux')
        with open(c_file_name, 'w') as c_file:
            c_file.write('#include <%s>\n' % header)
        fns = set()
        cmd = ('%s -c %s -o /dev/null -aux-info %s'
               % (cc_all, c_file_name, aux_file_name))
        try:
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError:
            return fns
        with open(aux_file_name, 'r') as aux_file:
            for line in aux_file:
                line = re.sub(r'/\*.*?\*/', '', line)
                line = line.strip()
                if line:
                    # The word before a '(' that isn't '(*' is the
                    # function name before the argument list (not
                    # fully general, but sufficient for -aux-info
                    # output on standard headers).
                    m = re.search(r'([A-Za-z0-9_]+) *\([^*]', line)
                    if m:
                        fns.add(m.group(1))
                    else:
                        raise ValueError("couldn't parse -aux-info output: %s"
                                         % line)
    return fns

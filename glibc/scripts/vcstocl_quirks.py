# VCSToChangeLog Quirks for the GNU C Library.

# Copyright (C) 2019-2021 Free Software Foundation, Inc.
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
# <http://www.gnu.org/licenses/>.

from frontend_c import Frontend
from projectquirks import ProjectQuirks
import re

class GlibcProjectQuirks(ProjectQuirks):
    repo = 'git'

    IGNORE_LIST = [
        'ChangeLog',
        'sysdeps/x86_64/dl-trampoline.h'
    ]

    MACRO_QUIRKS = \
        [{'orig': r'ElfW\((\w+)\)', 'sub': r'\1__ELF_NATIVE_CLASS_t'},
         {'orig': r'(libc_freeres_fn)\s*\((\w+)\)', 'sub': r'static void \1__\2 (void)'},
         {'orig': r'(IMPL)\s*\((\w+), .*\)$', 'sub': r'static void \1__\2 (void) {}'},
         {'orig': r'__(BEGIN|END)_DECLS', 'sub': r''},
         {'orig': 'weak_function', 'sub': '__attribute__ ((weak))'},
         {'orig': r'ATTRIBUTE_(CONST|MALLOC|PURE|FORMAT)',
          'sub': r'__attribute__ ((\1))'},
         {'orig': r'__THROW', 'sub': r'__attribute__ ((__nothrow__ __LEAF))'},
         {'orig': r'__THROWNL', 'sub': r'__attribute__ ((__nothrow__))'},
         {'orig': r'__nonnull \(\(([^)]+)\)\)',
          'sub': r'__attribute__ ((__nonnull__ \1))'},
         {'orig': r'([^_])attribute_(\w+)', 'sub': r'\1__attribute__ ((\2))'},
         {'orig': r'^attribute_(\w+)', 'sub': r'__attribute__ ((\1))'}]

    def __init__(self, debug):
        self.debug = debug
        ''' Build a list of macro calls used for symbol versioning and attributes.

        glibc uses a set of macro calls that do not end with a semi-colon and hence
        breaks our parser.  Identify those calls from include/libc-symbols.h and
        filter them out.
        '''
        with open('include/libc-symbols.h') as macrofile:
            op = macrofile.readlines()
            op = Frontend.remove_comments(self, op)
            self.C_MACROS = [re.sub(r'.*define (\w+).*', r'\1', x[:-1]) for x in op \
                             if 'define ' in x]

        super().__init__()

def get_project_quirks(debug):
    ''' Accessor function.
    '''
    return GlibcProjectQuirks(debug)

#!/usr/bin/python3
# Extract information from C headers.
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


def compute_c_consts(sym_data, cc):
    """Compute the values of some C constants.

    The first argument is a list whose elements are either strings
    (preprocessor directives, or the special string 'START' to
    indicate this function should insert its initial boilerplate text
    in the output there) or pairs of strings (a name and a C
    expression for the corresponding value).  Preprocessor directives
    in the middle of the list may be used to select which constants
    end up being evaluated using which expressions.

    """
    out_lines = []
    for arg in sym_data:
        if isinstance(arg, str):
            if arg == 'START':
                out_lines.append('void\ndummy (void)\n{')
            else:
                out_lines.append(arg)
            continue
        name = arg[0]
        value = arg[1]
        out_lines.append('asm ("@@@name@@@%s@@@value@@@%%0@@@end@@@" '
                         ': : \"i\" ((long int) (%s)));'
                         % (name, value))
    out_lines.append('}')
    out_lines.append('')
    out_text = '\n'.join(out_lines)
    with tempfile.TemporaryDirectory() as temp_dir:
        c_file_name = os.path.join(temp_dir, 'test.c')
        s_file_name = os.path.join(temp_dir, 'test.s')
        with open(c_file_name, 'w') as c_file:
            c_file.write(out_text)
        # Compilation has to be from stdin to avoid the temporary file
        # name being written into the generated dependencies.
        cmd = ('%s -S -o %s -x c - < %s' % (cc, s_file_name, c_file_name))
        subprocess.check_call(cmd, shell=True)
        consts = {}
        with open(s_file_name, 'r') as s_file:
            for line in s_file:
                match = re.search('@@@name@@@([^@]*)'
                                  '@@@value@@@[^0-9Xxa-fA-F-]*'
                                  '([0-9Xxa-fA-F-]+).*@@@end@@@', line)
                if match:
                    if (match.group(1) in consts
                        and match.group(2) != consts[match.group(1)]):
                        raise ValueError('duplicate constant %s'
                                         % match.group(1))
                    consts[match.group(1)] = match.group(2)
        return consts


def list_macros(source_text, cc):
    """List the preprocessor macros defined by the given source code.

    The return value is a pair of dicts, the first one mapping macro
    names to their expansions and the second one mapping macro names
    to lists of their arguments, or to None for object-like macros.

    """
    with tempfile.TemporaryDirectory() as temp_dir:
        c_file_name = os.path.join(temp_dir, 'test.c')
        i_file_name = os.path.join(temp_dir, 'test.i')
        with open(c_file_name, 'w') as c_file:
            c_file.write(source_text)
        cmd = ('%s -E -dM -o %s %s' % (cc, i_file_name, c_file_name))
        subprocess.check_call(cmd, shell=True)
        macros_exp = {}
        macros_args = {}
        with open(i_file_name, 'r') as i_file:
            for line in i_file:
                match = re.fullmatch('#define ([0-9A-Za-z_]+)(.*)\n', line)
                if not match:
                    raise ValueError('bad -dM output line: %s' % line)
                name = match.group(1)
                value = match.group(2)
                if value.startswith(' '):
                    value = value[1:]
                    args = None
                elif value.startswith('('):
                    match = re.fullmatch(r'\((.*?)\) (.*)', value)
                    if not match:
                        raise ValueError('bad -dM output line: %s' % line)
                    args = match.group(1).split(',')
                    value = match.group(2)
                else:
                    raise ValueError('bad -dM output line: %s' % line)
                if name in macros_exp:
                    raise ValueError('duplicate macro: %s' % line)
                macros_exp[name] = value
                macros_args[name] = args
    return macros_exp, macros_args


def compute_macro_consts(source_text, cc, macro_re, exclude_re=None):
    """Compute the integer constant values of macros defined by source_text.

    Macros must match the regular expression macro_re, and if
    exclude_re is defined they must not match exclude_re.  Values are
    computed with compute_c_consts.

    """
    macros_exp, macros_args = list_macros(source_text, cc)
    macros_set = {m for m in macros_exp
                  if (macros_args[m] is None
                      and re.fullmatch(macro_re, m)
                      and (exclude_re is None
                           or not re.fullmatch(exclude_re, m)))}
    sym_data = [source_text, 'START']
    sym_data.extend(sorted((m, m) for m in macros_set))
    return compute_c_consts(sym_data, cc)


def compare_macro_consts(source_1, source_2, cc, macro_re, exclude_re=None,
                         allow_extra_1=False, allow_extra_2=False):
    """Compare the values of macros defined by two different sources.

    The sources would typically be includes of a glibc header and a
    kernel header.  If allow_extra_1, the first source may define
    extra macros (typically if the kernel headers are older than the
    version glibc has taken definitions from); if allow_extra_2, the
    second source may define extra macros (typically if the kernel
    headers are newer than the version glibc has taken definitions
    from).  Return 1 if there were any differences other than those
    allowed, 0 if the macro values were the same apart from any
    allowed differences.

    """
    macros_1 = compute_macro_consts(source_1, cc, macro_re, exclude_re)
    macros_2 = compute_macro_consts(source_2, cc, macro_re, exclude_re)
    if macros_1 == macros_2:
        return 0
    print('First source:\n%s\n' % source_1)
    print('Second source:\n%s\n' % source_2)
    ret = 0
    for name, value in sorted(macros_1.items()):
        if name not in macros_2:
            print('Only in first source: %s' % name)
            if not allow_extra_1:
                ret = 1
        elif macros_1[name] != macros_2[name]:
            print('Different values for %s: %s != %s'
                  % (name, macros_1[name], macros_2[name]))
            ret = 1
    for name in sorted(macros_2.keys()):
        if name not in macros_1:
            print('Only in second source: %s' % name)
            if not allow_extra_2:
                ret = 1
    return ret

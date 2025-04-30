#!/usr/bin/python3
# Generate the locale/C-translit.h file.
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

import re
import sys


class StringLiteral:
    "Source of a string literal and its decomposition into code points."
    def __init__(self, s):
        # States:
        #  0 regular character sequence
        #  1 backslash seen
        #  2 in hexadecimal escape sequence
        state = 0
        result = []
        for ch in s:
            if state == 0:
                if ch == '\\':
                    state = 1
                else:
                    result.append(ord(ch))
            elif state == 1:
                if ch in "\\\"":
                    result.append(ord(ch))
                    state = 0
                elif ch == 'x':
                    state = 2
                    result.append(0)
                else:
                    raise ValueError("invalid character {!r} in {!r}".format(
                        ch, s))
            elif state == 2:
                if ch in "0123456789abcdefABCDEF":
                    result[-1] = result[-1] * 16 + int(ch, 16)
                else:
                    if ch == '\\':
                        state = 1
                    else:
                        state = 0
        if state == 1:
            raise ValueError("trailing backslash in {!r}".format(s))

        self.source = s
        self.decoded = tuple(result)


class Translit:
    "Pair of transliteration and source."

    __RE_TRANSLIT = re.compile(
        r'^"((?:[^"\\]|\\x[0-9a-fA-F])+)"\s+'
        r'"((?:[^"\\]|\\["\\])*)"\s*(?:#.*)?$')

    def __init__(self, line):
        match = self.__RE_TRANSLIT.match(line)
        if not match:
            raise IOError("invalid line {}: {!r}".format(
                lineno + 1, line))
        codepoints, replacement = match.groups()
        self.codepoints = StringLiteral(codepoints)
        self.replacement = StringLiteral(replacement)


# List of Translit objects.
translits = []

# Read transliterations from standard input.
for lineno, line in enumerate(sys.stdin):
    line = line.strip()
    # Skip empty lines and comments.
    if (not line) or line[0] == '#':
        continue
    translit = Translit(line)
    # Check ordering of codepoints.
    if translits \
       and translit.codepoints.decoded <= translits[-1].codepoints.decoded:
        raise IOError("unexpected codepoint {!r} on line {}: {!r}".format(
            translit.codepoints.decoded, lineno + 1, line))
    translits.append(translit)

# Generate the C sources.
write = sys.stdout.write
write("#include <stdint.h>\n")
write("#define NTRANSLIT {}\n".format(len(translits)))

write("static const uint32_t translit_from_idx[] =\n{\n  ")
col = 2
total = 0
for translit in translits:
    if total > 0:
        if col + 7 >= 79:
            write(",\n  ")
            col = 2
        else:
            write(", ")
            col += 2
    write("{:4}".format(total))
    total += len(translit.codepoints.decoded) + 1
    col += 4
write("\n};\n")

write("static const wchar_t translit_from_tbl[] =\n ")
col = 1
first = True
for translit in translits:
    if first:
        first = False
    else:
        if col + 6 >= 79:
            write("\n ")
            col = 1
        write(" L\"\\0\"")
        col += 6
    if col > 2 and col + len(translit.codepoints.source) + 4 >= 79:
        write("\n  ")
        col = 2
    else:
        write(" ")
        col += 1
    write("L\"{}\"".format(translit.codepoints.source))
    col += len(translit.codepoints.source) + 3
write(";\n")

write("static const uint32_t translit_to_idx[] =\n{\n  ")
col = 2
total = 0
for translit in translits:
    if total > 0:
        if col + 7 >= 79:
            write(",\n  ")
            col = 2
        else:
            write(", ")
            col += 2
    write("{:4}".format(total))
    total += len(translit.replacement.decoded) + 2
    col += 4
write("\n};\n")

write("static const wchar_t translit_to_tbl[] =\n ")
col = 1
first = True
for translit in translits:
    if first:
        first = False
    else:
        if col + 6 >= 79:
            write("\n ")
            col = 1
        write(" L\"\\0\"")
        col += 6
    if col > 2 and col + len(translit.replacement.source) + 6 >= 79:
        write("\n  ")
        col = 2
    else:
        write(" ")
        col += 1
    write("L\"{}\\0\"".format(translit.replacement.source))
    col += len(translit.replacement.source) + 5
write(";\n")

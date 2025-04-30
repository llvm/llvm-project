#!/usr/bin/python3
# Generate DNS RR type constants for resolv header files.
# Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

"""Generate DNS RR type constants for resolv header files.

resolv/arpa/nameser.h and resolv/arpa/nameser_compat.h contain lists
of RR type constants.  This script downloads the current definitions
from the IANA DNS Parameters protocol registry and translates it into
the two different lists.

Two lists are written to standard output.  The first one contains enum
constants for resolv/arpa/nameser.h.  The second one lists the
preprocessor macros for resolv/arpa/nameser_compat.h.

"""

# URL of the IANA registry.
source = "http://www.iana.org/assignments/dns-parameters/dns-parameters-4.csv"

import collections
import csv
import io
import urllib.request

Type = collections.namedtuple("Type", "name number comment")

def get_types(source):
    for row in csv.reader(io.TextIOWrapper(urllib.request.urlopen(source))):
        if row[0] in ('TYPE', 'Unassigned', 'Private use', 'Reserved'):
            continue
        name, number, comment = row[:3]
        if name == '*':
            name = 'ANY'
            comment = 'request for all cached records'
        number = int(number)
        yield Type(name, number, comment)

types = list(get_types(source))

print("// enum constants for resolv/arpa/nameser.h")
print()
for typ in types:
    name = typ.name.replace("-", "_").lower()
    print("    ns_t_{0} = {1.number},".format(name, typ))
print()

print("// macro aliases resolv/arpa/nameser_compat.h")
print()
for typ in types:
    name = typ.name.replace("-", "_")
    print("#define T_{0} ns_t_{1}".format(name.upper(), name.lower()))
print()

#!/bin/sh -f
#
# Copyright (C) 1998-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library and contains tests for
# the rpmatch(3)-implementation.
# contributed by Jochen Hein <jochen.hein@delphi.central.de>

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <https://www.gnu.org/licenses/>.

set -e

common_objpfx=$1
tst_rpmatch=$2

rc=0
while IFS=\& read locale string result dummy; do
    if [ "$locale" != "#" ]; then
	${tst_rpmatch} $locale $string $result < /dev/null \
	|| { echo "$locale $string $result  FAILED"; exit 1; }
    fi
done <<EOF
#& These are the tests for rpmatch in glibc.  Each line contains one test,
#& comments start with #& in the first column.  The fields are separated
#& by ampersand signs and contain: the locale, the string, the expected
#& return value of rpmatch(3).  If the test fails, test-rpmatch prints
#& all these informations
C&Yes&1
C&yes&1
C&YES&1
C&YeS&1
C&YEs&1
C&yEs&1
C&yES&1
C&yeS&1
C&No&0
C&no&0
#& Uh, that's nonsense
C&nonsens&0
C&Error&-1
de_DE.ISO-8859-1&Yes&1
de_DE.ISO-8859-1&Ja&1
de_DE.ISO-8859-1&Jammerschade&1
de_DE.ISO-8859-1&dejavu&-1
de_DE.ISO-8859-1&Nein&0
de_DE.ISO-8859-1&Fehler&-1
de_DE.ISO-8859-1&jein&1
EOF

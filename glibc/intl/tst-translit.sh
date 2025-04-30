#!/bin/sh
# Test of transliteration in gettext functions.
# Copyright (C) 2000-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.

# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

set -e

common_objpfx=$1
test_program_prefix=$2
objpfx=$3

# Create the locale directories.
mkdir -p ${objpfx}localedir/existing-locale/LC_MESSAGES

msgfmt -o ${objpfx}domaindir/existing-locale/LC_MESSAGES/translit.mo \
       translit.po

${test_program_prefix} \
${objpfx}tst-translit > ${objpfx}tst-translit.out ${objpfx}domaindir

exit $?

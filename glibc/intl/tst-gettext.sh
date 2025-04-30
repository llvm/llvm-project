#!/bin/sh
# Test of gettext functions.
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
test_program_prefix_before_env=$2
run_program_env=$3
test_program_prefix_after_env=$4
objpfx=$5
malloc_trace=$6

# Generate the test data.

# Create the locale directories.
mkdir -p ${objpfx}localedir/existing-locale/LC_MESSAGES
for f in ADDRESS COLLATE CTYPE IDENTIFICATION MEASUREMENT MONETARY NAME NUMERIC PAPER TELEPHONE TIME; do
  cp -f ${common_objpfx}localedata/de_DE.UTF-8/LC_$f \
        ${objpfx}localedir/existing-locale
done
cp -f ${common_objpfx}localedata/de_DE.UTF-8/LC_MESSAGES/SYS_LC_MESSAGES \
      ${objpfx}localedir/existing-locale/LC_MESSAGES

# Create the domain directories.
mkdir -p ${objpfx}domaindir/existing-locale/LC_MESSAGES
mkdir -p ${objpfx}domaindir/existing-locale/LC_TIME
# Populate them.
msgfmt -o ${objpfx}domaindir/existing-locale/LC_MESSAGES/existing-domain.mo \
       -f ${objpfx}tst-gettext-de.po
msgfmt -o ${objpfx}domaindir/existing-locale/LC_TIME/existing-time-domain.mo \
       -f ${objpfx}tst-gettext-de.po

# Now run the test.
${test_program_prefix_before_env} \
${run_program_env} \
MALLOC_TRACE=$malloc_trace \
LD_PRELOAD=${common_objpfx}malloc/libc_malloc_debug.so \
LOCPATH=${objpfx}localedir:${common_objpfx}localedata \
${test_program_prefix_after_env} \
${objpfx}tst-gettext > ${objpfx}tst-gettext.out ${objpfx}domaindir

exit $?

#!/bin/sh
# Copyright (C) 1999-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
# Contributed by Andreas Jaeger <aj@arthur.rhein-neckar.de>, 1999.

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

common_objpfx=$1; shift
test_program_prefix=$1; shift

testout=${common_objpfx}/grp/tst_fgetgrent.out

result=0

${test_program_prefix} \
${common_objpfx}grp/tst_fgetgrent 0 > ${testout} || result=1

${test_program_prefix} \
${common_objpfx}grp/tst_fgetgrent 1 >> ${testout} || result=1

${test_program_prefix} \
${common_objpfx}grp/tst_fgetgrent 2 >> ${testout} || result=1

${test_program_prefix} \
${common_objpfx}grp/tst_fgetgrent 3 >> ${testout} || result=1

exit $result

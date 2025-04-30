#!/bin/sh
# Test character mapping definitions.
# Copyright (C) 1999-2021 Free Software Foundation, Inc.
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
run_program_prefix_before_env=$2
run_program_env=$3
run_program_prefix_after_env=$4
test_program_prefix_before_env=$5
test_program_prefix_after_env=$6

# Generate the necessary locale data.
${run_program_prefix_before_env} \
${run_program_env} \
I18NPATH=. \
${run_program_prefix_after_env} \
${common_objpfx}locale/localedef --quiet \
-i tests/trans.def -f charmaps/ISO-8859-1 \
${common_objpfx}localedata/tt_TT ||
exit 1

# Run the test program.
${test_program_prefix_before_env} \
${run_program_env} \
LC_ALL=tt_TT ${test_program_prefix_after_env} \
${common_objpfx}localedata/tst-trans > ${common_objpfx}localedata/tst-trans.out

exit $?

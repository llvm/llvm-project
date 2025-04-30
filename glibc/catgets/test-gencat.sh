#!/bin/sh
# Test escape character handling in gencat.
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
test_program_cmd_before_env=$2
run_program_env=$3
test_program_cmd_after_env=$4

# Run the test program.
${test_program_cmd_before_env} \
  ${run_program_env} \
  NLSPATH=${common_objpfx}catgets/%N.%c.cat LC_ALL=ja_JP.SJIS \
  ${test_program_cmd_after_env} \
    > ${common_objpfx}catgets/test-gencat.out

# Compare with the expected result.
cmp - ${common_objpfx}catgets/test-gencat.out <<"EOF"
LC_MESSAGES = ja_JP.SJIS
sample1:ABCDEF:
sample2:日本語:
sample3:予定表:
sample4:TEST	TAB:
sample5:機能	十種類:
double slash\
another line
EOF
res=$?

cat <<EOF |
#define AnotherSet 0x2	/* *standard input*:13 */
#define AnotherFOO 0x1	/* *standard input*:14 */
EOF
cmp ${common_objpfx}catgets/test-gencat.h - || res=1

exit $res

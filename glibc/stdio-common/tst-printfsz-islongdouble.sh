#!/bin/sh
# Test for the behaviour of 'is_binary128' in printf_size.
# Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

test_program=$1; shift
test_program_prefix=$1; shift
test_program_output=$1; shift

status=0

${test_program_prefix} \
  ${test_program} \
  > ${test_program_output} || status=1

echo -n "2k4k" | cmp - ${test_program_output} > /dev/null 2>&1 ||
{
  status=1
  echo "*** output comparison failed"
}

exit $status

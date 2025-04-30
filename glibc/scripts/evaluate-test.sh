#!/bin/sh
# Output a test status line.
# Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

# usage: evaluate-test.sh test_name rc xfail stop_on_failure

test_name=$1
rc=$2
orig_rc=$rc
xfail=$3
stop_on_failure=$4

if [ $rc -eq 77 ]; then
  result="UNSUPPORTED"
  rc=0
else
  if [ $rc -eq 0 ]; then
    result="PASS"
  else
    result="FAIL"
  fi

  if $xfail; then
    result="X$result"
    rc=0
  fi
fi

echo "$result: $test_name"
echo "original exit status $orig_rc"
if $stop_on_failure; then
  exit $rc
else
  exit 0
fi

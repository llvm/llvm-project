#!/bin/sh
# Bug 18125: Test the exit functionality of setcontext().
# Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

test_pre="${test_program_prefix_before_env} ${run_program_env}"
test="${test_program_prefix_after_env} ${objpfx}tst-setcontext3"
out=${objpfx}tst-setcontext3.out

cleanup() {
  rm -f $tempfile
}
trap cleanup 0

tempfile=$(mktemp "${objpfx}tst-setcontext3.XXXXXXXXXX")

# We want to run the test program and see if secontext called
# exit() and wrote out the test file we specified.  If the
# test exits with a non-zero status this will fail because we
# are using `set -e`.
$test_pre $test "$tempfile"

# Look for resulting file.
if [ -e "$tempfile" ]; then
  echo "PASS: tst-setcontext3 an exit() and created $tempfile"
  exit 0
else
  echo "FAIL: tst-setcontext3 did not create $tempfile"
  exit 1
fi

#!/bin/sh
# Test failure recording (with and without --direct).
# Copyright (C) 2016-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.

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
# <https://www.gnu.org/licenses/>.  */

set -e

common_objpfx=$1; shift
test_program_prefix_before_env=$1; shift
run_program_env=$1; shift
test_program_prefix_after_env=$1; shift

run_test () {
    expected_status="$1"
    expected_output="$2"
    shift 2
    args="${common_objpfx}support/tst-support_record_failure $*"
    echo "running: $args"
    set +e
    output="$(${test_program_prefix_before_env} \
		 ${run_program} ${test_program_prefix_after_env} $args)"
    status=$?
    set -e
    echo "  exit status: $status"
    if test "$output" != "$expected_output" ; then
	echo "error: unexpected output: $output"
	exit 1
    fi
    if test "$status" -ne "$expected_status" ; then
	echo "error: exit status $expected_status expected"
	exit 1
    fi
}

different_status () {
    direct="$1"
    run_test 1 "error: 1 test failures" $direct --status=0
    run_test 1 "error: 1 test failures" $direct --status=1
    run_test 2 "error: 1 test failures" $direct --status=2
    run_test 1 "error: 1 test failures" $direct --status=77
    run_test 2 "error: tst-support_record_failure.c:109: not true: false
error: 1 test failures" $direct --test-verify
    run_test 2 "error: tst-support_record_failure.c:109: not true: false
info: execution passed failed TEST_VERIFY
error: 1 test failures" $direct --test-verify --verbose
}

different_status
different_status --direct

run_test 1 "error: tst-support_record_failure.c:116: not true: false
error: 1 test failures" --test-verify-exit
# --direct does not print the summary error message if exit is called.
run_test 1 "error: tst-support_record_failure.c:116: not true: false" \
	 --direct --test-verify-exit

#!/bin/sh
# Test for wordexp(3).
# Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

# Some of these tests may look a little weird.
# The first parameter to wordexp-test is what it gives to wordexp.
# The others are just there to be parameters.

common_objpfx=$1; shift
test_program_prefix_before_env=$1; shift
run_program_env=$1; shift
test_program_prefix_after_env=$1; shift
logfile=${common_objpfx}posix/wordexp-tst.out
testout=${common_objpfx}posix/wordexp-test-result

result=0
rm -f $logfile
# This is written in this funny way so that there is no trailing whitespace.
# The first line contains a space followed by a tab.
IFS=" 	\

"
export IFS

failed=0
${test_program_prefix_before_env} ${run_program_env} IFS="$IFS" \
${test_program_prefix_after_env} \
${common_objpfx}posix/wordexp-test '$*' > ${testout}1
cat <<"EOF" | cmp - ${testout}1 >> $logfile || failed=1
wordexp returned 0
we_wordv[0] = "$*"
EOF
if test $failed -ne 0; then
  echo '$* test failed'
  status=1
fi

failed=0
${test_program_prefix_before_env} ${run_program_env} IFS="$IFS" \
${test_program_prefix_after_env} \
${common_objpfx}posix/wordexp-test '${*}' unquoted > ${testout}2
cat <<"EOF" | cmp - ${testout}2 >> $logfile || failed=1
wordexp returned 0
we_wordv[0] = "${*}"
we_wordv[1] = "unquoted"
EOF
if test $failed -ne 0; then
  echo '${*} test failed'
  status=1
fi

failed=0
${test_program_prefix_before_env} ${run_program_env} IFS="$IFS" \
${test_program_prefix_after_env} \
${common_objpfx}posix/wordexp-test '$@' unquoted > ${testout}3
cat <<"EOF" | cmp - ${testout}3 >> $logfile || failed=1
wordexp returned 0
we_wordv[0] = "$@"
we_wordv[1] = "unquoted"
EOF
if test $failed -ne 0; then
  echo '$@ test failed'
  status=1
fi

failed=0
${test_program_prefix_before_env} ${run_program_env} IFS="$IFS" \
${test_program_prefix_after_env} \
${common_objpfx}posix/wordexp-test '"$* quoted"' param > ${testout}4
cat <<"EOF" | cmp - ${testout}4 >> $logfile || failed=1
wordexp returned 0
we_wordv[0] = ""$* quoted" param quoted"
EOF
if test $failed -ne 0; then
  echo '$* quoted test failed'
  status=1
fi

failed=0
${test_program_prefix_before_env} ${run_program_env} IFS="$IFS" \
${test_program_prefix_after_env} \
${common_objpfx}posix/wordexp-test '"$@ quoted"' param > ${testout}5
cat <<"EOF" | cmp - ${testout}5 >> $logfile || failed=1
wordexp returned 0
we_wordv[0] = ""$@ quoted""
we_wordv[1] = "param quoted"
EOF
if test $failed -ne 0; then
  echo '$@ quoted test failed'
  status=1
fi
# Why?  Because bash does it that way..

failed=0
${test_program_prefix_before_env} ${run_program_env} IFS="$IFS" \
${test_program_prefix_after_env} \
${common_objpfx}posix/wordexp-test '$#' 2 3 4 5 > ${testout}6
cat <<"EOF" | cmp - ${testout}6 >> $logfile || failed=1
wordexp returned 0
we_wordv[0] = "5"
EOF
if test $failed -ne 0; then
  echo '$# test failed'
  status=1
fi

failed=0
${test_program_prefix_before_env} ${run_program_env} IFS="$IFS" \
${test_program_prefix_after_env} \
${common_objpfx}posix/wordexp-test '$2 ${3} $4' 2nd 3rd "4 th" > ${testout}7
cat <<"EOF" | cmp - ${testout}7 >> $logfile || failed=1
wordexp returned 0
we_wordv[0] = "2nd"
we_wordv[1] = "3rd"
we_wordv[2] = "4"
we_wordv[3] = "th"
EOF
if test $failed -ne 0; then
  echo '$2 ${3} $4 test failed'
  status=1
fi

failed=0
${test_program_prefix_before_env} ${run_program_env} IFS="$IFS" \
${test_program_prefix_after_env} \
${common_objpfx}posix/wordexp-test '${11}' 2 3 4 5 6 7 8 9 10 11 > ${testout}8
cat <<"EOF" | cmp - ${testout}8 >> $logfile || failed=1
wordexp returned 0
we_wordv[0] = "11"
EOF
if test $failed -ne 0; then
  echo '${11} test failed'
  status=1
fi

failed=0
${test_program_prefix_before_env} ${run_program_env} IFS="$IFS" \
${test_program_prefix_after_env} \
${common_objpfx}posix/wordexp-test '"a $@ b"' c d > ${testout}9
cat <<"EOF" | cmp - ${testout}9 >> $logfile || failed=1
wordexp returned 0
we_wordv[0] = "a "a $@ b""
we_wordv[1] = "c"
we_wordv[2] = "d b"
EOF
if test $failed -ne 0; then
  echo '"a $@ b" test failed'
  status=1
fi

${test_program_prefix_before_env} ${run_program_env} IFS="$IFS" \
${test_program_prefix_after_env} \
${common_objpfx}posix/wordexp-test '${#@} ${#2} *$**' two 3 4 > ${testout}10
cat <<"EOF" | cmp - ${testout}10 || failed=1
wordexp returned 0
we_wordv[0] = "4"
we_wordv[1] = "3"
we_wordv[2] = "*${#@}"
we_wordv[3] = "${#2}"
we_wordv[4] = "*$**"
we_wordv[5] = "two"
we_wordv[6] = "3"
we_wordv[7] = "4*"
EOF

exit $result

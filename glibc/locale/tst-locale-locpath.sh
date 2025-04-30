#!/bin/sh
# Test that locale prints LOCPATH on failure.
# Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

set -ex

common_objpfx=$1
test_wrapper_env=$2
run_program_env=$3

LIBPATH="$common_objpfx"

testroot="${common_objpfx}locale/tst-locale-locpath-directory"
cleanup () {
    rm -rf "$testroot"
}
trap cleanup 0

rm -rf "$testroot"
mkdir -p $testroot

${test_wrapper_env} \
${run_program_env} LANG= LC_ALL=invalid-locale LOCPATH=does-not-exist \
${common_objpfx}elf/ld.so --library-path "$LIBPATH" \
  "${common_objpfx}locale/locale" \
  > "$testroot/stdout" 2> "$testroot/stderr"

echo "* standard error"
cat "$testroot/stderr"
echo "* standard output"
cat "$testroot/stdout"

cat > "$testroot/stderr-expected" <<EOF
${common_objpfx}locale/locale: Cannot set LC_CTYPE to default locale: No such file or directory
${common_objpfx}locale/locale: Cannot set LC_MESSAGES to default locale: No such file or directory
${common_objpfx}locale/locale: Cannot set LC_ALL to default locale: No such file or directory
warning: The LOCPATH variable is set to "does-not-exist"
EOF

cat > "$testroot/stdout-expected" <<EOF
LANG=
LC_CTYPE="invalid-locale"
LC_NUMERIC="invalid-locale"
LC_TIME="invalid-locale"
LC_COLLATE="invalid-locale"
LC_MONETARY="invalid-locale"
LC_MESSAGES="invalid-locale"
LC_PAPER="invalid-locale"
LC_NAME="invalid-locale"
LC_ADDRESS="invalid-locale"
LC_TELEPHONE="invalid-locale"
LC_MEASUREMENT="invalid-locale"
LC_IDENTIFICATION="invalid-locale"
LC_ALL=invalid-locale
EOF

errors=0
if ! cmp -s "$testroot/stderr-expected" "$testroot/stderr" ; then
    echo "error: standard error not correct"
    errors=1
fi
if ! cmp -s "$testroot/stdout-expected" "$testroot/stdout" ; then
    echo "error: standard output not correct"
    errors=1
fi
exit $errors

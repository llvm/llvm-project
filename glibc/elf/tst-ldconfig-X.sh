#!/bin/sh
# Test that ldconfig -X does not remove stale symbolic links.
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

set -ex

common_objpfx=$1
test_wrapper_env=$2
run_program_env=$3

testroot="${common_objpfx}elf/bug19610-test-directory"
cleanup () {
    rm -rf "$testroot"
}
trap cleanup 0

rm -rf "$testroot"
mkdir -p $testroot/lib $testroot/etc

# Relative symbolic link target.
ln -s libdoesnotexist.so.1.1 $testroot/lib/libdoesnotexist.so.1

# Absolute symbolic link target.
ln -s $testroot/opt/sw/lib/libdoesnotexist2.so.1.1 $testroot/lib/

errors=0
check_files () {
    for name in libdoesnotexist.so.1 libdoesnotexist2.so.1.1 ; do
	path="$testroot/lib/$name"
	if test ! -h $path ; then
	    echo "error: missing file: $path"
	    errors=1
	fi
    done
}

check_files

${test_wrapper_env} \
${run_program_env} \
${common_objpfx}elf/ldconfig -X -f /dev/null \
  -C $testroot/etc/ld.so.cache \
  $testroot/lib

check_files

exit $errors

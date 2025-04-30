#!/bin/sh
# Test how rtld loads itself.
# Copyright (C) 2012-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
#

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

rtld=$1
test_wrapper=$2
test_wrapper_env=$3
result=0

echo '# normal mode'
${test_wrapper} $rtld $rtld 2>&1 && rc=0 || rc=$?
echo "# exit status $rc"
test $rc -le 127 || result=1

echo '# list mode'
${test_wrapper} $rtld --list $rtld 2>&1 && rc=0 || rc=$?
echo "# exit status $rc"
test $rc -eq 0 || result=1

echo '# verify mode'
${test_wrapper} $rtld --verify $rtld 2>&1 && rc=0 || rc=$?
echo "# exit status $rc"
test $rc -eq 2 || result=1

echo '# trace mode'
${test_wrapper_env} LD_TRACE_LOADED_OBJECTS=1 \
    $rtld $rtld 2>&1 && rc=0 || rc=$?
echo "# exit status $rc"
test $rc -eq 0 || result=1

exit $result

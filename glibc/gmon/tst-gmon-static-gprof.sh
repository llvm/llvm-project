#!/bin/sh
# Check the output of gprof against a carfully crafted static binary.
# Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

LC_ALL=C
export LC_ALL
set -e
exec 2>&1

GPROF="$1"
program="$2"
data="$3"

actual=$(mktemp)
expected=$(mktemp)
expected_dot=$(mktemp)
cleanup () {
    rm -f "$actual"
    rm -f "$expected"
    rm -f "$expected_dot"
}
trap cleanup 0

cat > "$expected" <<EOF
f1 2000
f2 1000
main 1
EOF

# Special version for powerpc with function descriptors.
cat > "$expected_dot" <<EOF
.f1 2000
.f2 1000
.main 1
EOF

"$GPROF" -C "$program" "$data" \
    | awk -F  '[(): ]' '/executions/{print $5, $8}' \
    | sort > "$actual"

if cmp -s "$actual" "$expected_dot" \
   || diff -u --label expected "$expected" --label actual "$actual" ; then
    echo "PASS"
else
    echo "FAIL"
    exit 1
fi

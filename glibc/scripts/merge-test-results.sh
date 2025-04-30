#!/bin/sh
# Merge test results of individual tests or subdirectories.
# Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

# usage: merge-test-results.sh -s objpfx subdir test-name...
# (subdirectory tests; empty subdir at top level), or
#        merge-test-results.sh -t objpfx subdir-file-name subdir...
# (top-level merge)

set -e

type=$1
objpfx=$2
shift 2

case $type in
  -s)
    subdir=$1
    shift
    subdir=${subdir:+$subdir/}
    for t in "$@"; do
      if [ -s "$objpfx$t.test-result" ]; then
	# This loop is called thousands of times even when there's
	# nothing to do.  Avoid using non-built-in commands (like
	# /bin/head) where possible.  We assume "echo" is typically a
	# built-in.
	IFS= read -r line < "$objpfx$t.test-result"
	echo "$line"
      else
	echo "UNRESOLVED: $subdir$t"
      fi
    done
    ;;

  -t)
    subdir_file_name=$1
    shift
    for d in "$@"; do
      if [ -f "$objpfx$d/$subdir_file_name" ]; then
	cat "$objpfx$d/$subdir_file_name"
      else
	echo "ERROR: test results for $d directory missing"
      fi
    done
    ;;

  *)
    echo "unknown type $type" >&2
    exit 1
    ;;
esac

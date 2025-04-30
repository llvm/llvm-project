#!/bin/sh
# Check the set of headers with conformtest expectations for a given standard.
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

set -e

std=$1
CC=$2
expected_list=$3
all_data_files=$4

new_list=

for f in $all_data_files; do
  h=${f#data/}
  h=${h%-data}
  exp=$($CC -D$std -x c -E $f | sed -e '/^#/d' -e '/^[ 	]*$/d')
  if [ "$exp" ]; then
    new_list="$new_list $h"
  fi
done

echo "Headers with expectations for $std: $new_list"
echo "Expected list: $expected_list"

rc=0

for h in $expected_list; do
  case " $new_list " in
    (*" $h "*)
      ;;
    (*)
      echo "Missing expectations for $h."
      rc=1
      ;;
  esac
done

for h in $new_list; do
  case " $expected_list " in
    (*" $h "*)
      ;;
    (*)
      echo "Spurious expectations for $h."
      rc=1
      ;;
  esac
done

exit $rc

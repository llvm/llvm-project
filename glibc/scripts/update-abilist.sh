#!/bin/sh
# Update abilist files based on differences on one architecture.
# Copyright (C) 2015-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
#
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
# <https://www.gnu.org/licenses/>.

set -e
export LC_ALL=C

if [ $# -lt 2 ]; then
  echo "usage: $0 OLD-FILE NEW-FILE FILES-TO-BE-PATCHED..." 1>&2
  exit 2
elif [ $# -eq 2 ]; then
  echo "info: no files to patch" 1>&2
  exit 0
fi

old_file="$1"
shift
new_file="$1"
shift

tmp_old_sorted="$(mktemp)"
tmp_new_sorted="$(mktemp)"
tmp_new_symbols="$(mktemp)"
tmp_patched="$(mktemp)"

cleanup () {
  rm -f -- "$tmp_old_sorted" "$tmp_new_sorted" \
    "$tmp_new_symbols" "$tmp_patched"
}

trap cleanup 0

sort -u -o "$tmp_old_sorted" -- "$old_file"
sort -u -o "$tmp_new_sorted" -- "$new_file"

# -1 skips symbols only in $old_file (deleted symbols).
# -3 skips symbols in both files (unchanged symbols).
comm -1 -3 "$tmp_old_sorted" "$tmp_new_sorted" > "$tmp_new_symbols"

new_symbol_count="$(wc -l < "$tmp_new_symbols")"
if [ "$new_symbol_count" -eq 0 ]; then
  echo "info: no symbols added" 1>&2
  exit 0
fi

echo "info: $new_symbol_count symbol(s) added" 1>&2

for to_be_patched in "$@" ; do
  sort -u -o "$tmp_patched" -- "$to_be_patched" "$tmp_new_symbols"
  if ! cmp -s -- "$to_be_patched" "$tmp_patched"; then
    echo "info: updating $to_be_patched" 1>&2
    cp -- "$tmp_patched" "$to_be_patched"
  fi
done

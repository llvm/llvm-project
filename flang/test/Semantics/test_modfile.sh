#!/usr/bin/env bash
# Compile a source file and compare generated .mod files against expected.

set -e
FLANG_FC1_OPTIONS="-fsyntax-only"
srcdir=$(dirname $0)
source $srcdir/common.sh

actual=$temp/actual.mod
expect=$temp/expect.mod
actual_files=$temp/actual_files
prev_files=$temp/prev_files
diffs=$temp/diffs

set $src

touch $actual
for src in "$@"; do
  [[ ! -f $src ]] && echo "File not found: $src" && exit 1
  path=$(git ls-files --full-name $src 2>/dev/null || echo $src)
  (
    cd $temp
    ls -1 *.mod > prev_files
    $FLANG_FC1 $FLANG_FC1_OPTIONS $src
    ls -1 *.mod | comm -13 prev_files -
  ) > $actual_files
  expected_files=$(sed -n 's/^!Expect: \(.*\)/\1/p' $src | sort)
  extra_files=$(echo "$expected_files" | comm -23 $actual_files -)
  if [[ ! -z "$extra_files" ]]; then
    echo "Unexpected .mod files produced:" $extra_files
    die FAIL $path
  fi
  for mod in $expected_files; do
    if [[ ! -f $temp/$mod ]]; then
      echo "Compilation did not produce expected mod file: $mod"
      die FAIL $path
    fi
    # The first three bytes of the file are a UTF-8 BOM
    sed '/^[^!]*!mod\$/d' $temp/$mod > $actual
    sed '1,/^!Expect: '"$mod"'/d' $src | sed -e '/^$/,$d' -e 's/^!//' > $expect
    if ! diff -w -U999999 $expect $actual > $diffs; then
      echo "Module file $mod differs from expected:"
      sed '1,2d' $diffs
      die FAIL $path
    fi
  done
  rm -f $actual $expect
done
echo PASS

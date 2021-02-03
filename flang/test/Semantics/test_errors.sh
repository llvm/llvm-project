#!/usr/bin/env bash
# Compile a source file and check errors against those listed in the file.
# Change the compiler by setting the F18 environment variable.

F18_OPTIONS="-fsyntax-only"
srcdir=$(dirname $0)
source $srcdir/common.sh
[[ ! -f $src ]] && die "File not found: $src"

log=$temp/log
actual=$temp/actual
expect=$temp/expect
diffs=$temp/diffs

cmd="$F18 $F18_OPTIONS $src"
( cd $temp; $cmd ) > $log 2>&1
if [[ $? -ge 128 ]]; then
  cat $log
  exit 1
fi

# $actual has errors from the compiler; $expect has them from !ERROR comments in source
# Format both as "<line>: <text>" so they can be diffed.
sed -n 's=^[^:]*:\([^:]*\):[^:]*: error: =\1: =p' $log > $actual
awk '
  BEGIN { FS = "!ERROR: "; }
  /^ *!ERROR: / { errors[nerrors++] = $2; next; }
  { for (i = 0; i < nerrors; ++i) printf "%d: %s\n", NR, errors[i]; nerrors = 0; }
' $src > $expect

if diff -U0 $actual $expect > $diffs; then
  echo PASS
else
  echo "$cmd"
  < $diffs \
    sed -n -e 's/^-\([0-9]\)/actual at \1/p' -e 's/^+\([0-9]\)/expect at \1/p' \
    | sort -n -k 3
  die FAIL
fi

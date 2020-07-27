#!/usr/bin/env bash
# Compile a source file with '-funparse-with-symbols' and verify
# we get the right symbols in the output, i.e. the output should be
# the same as the input, except for the copyright comment.
# Change the compiler by setting the F18 environment variable.

F18_OPTIONS="-funparse-with-symbols"
srcdir=$(dirname $0)
source $srcdir/common.sh
[[ ! -f $src ]] && echo "File not found: $src" && exit 1

src1=$temp/1.f90
src2=$temp/2.f90
src3=$temp/3.f90
diffs=$temp/diffs

# Strip out blank lines and all comments except "!DEF:", "!REF:", and "!$omp"
sed -e 's/!\([DR]EF:\)/KEEP \1/' -e 's/!\($omp\)/KEEP \1/' \
  -e 's/!\($acc\)/KEEP \1/' -e 's/!.*//' -e 's/ *$//' -e '/^$/d' \
  -e 's/KEEP \([DR]EF:\)/!\1/' -e 's/KEEP \($omp\)/!\1/' \
  -e 's/KEEP \($acc\)/!\1/' \
  $src > $src1
egrep -v '![DR]EF:' $src1 > $src2  # strip out DEF and REF comments
# compile, inserting comments for symbols:
( cd $temp; $F18 $F18_OPTIONS $(basename $src2) ) > $src3

if diff -w -U999999 $src1 $src3 > $diffs; then
  echo PASS
else
  sed '1,/^\@\@/d' $diffs
  echo
  echo FAIL
  exit 1
fi

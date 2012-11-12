#!/bin/bash
#
# This script produces a list of all diagnostics that are defined
# in Diagnostic*.td files but not used in sources.
#

ALL_DIAGS=$(mktemp)
ALL_SOURCES=$(mktemp)

grep -E --only-matching --no-filename '(err_|warn_|ext_|note_)[a-z_]+ ' ./include/clang/Basic/Diagnostic*.td > $ALL_DIAGS
find lib include tools -name \*.cpp -or -name \*.h > $ALL_SOURCES
for DIAG in $(cat $ALL_DIAGS); do
  if ! grep -r $DIAG $(cat $ALL_SOURCES) > /dev/null; then
    echo $DIAG
  fi;
done

rm $ALL_DIAGS $ALL_SOURCES


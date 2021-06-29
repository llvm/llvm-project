#!/usr/bin/env bash
set -e

if test -z "$PATHTOCLANG"; then
  PATHTOCLANG=clang
fi

for INDEX in $SOURCE_INDEXES
do
  CFLAGS=$(eval echo "\$COMPILER_OPTIONS_$INDEX")
  SRCFILE=$(eval echo "\$SOURCE_FILE_$INDEX")
  OBJFILE=$(eval echo "\$OBJECT_FILE_$INDEX")
  $PATHTOCLANG -std=gnu11 -c $CFLAGS $SRCFILE -o $OBJFILE
done

$PATHTOCLANG $LINKER_OPTIONS $OBJECT_FILES -o $EXECUTABLE_FILE

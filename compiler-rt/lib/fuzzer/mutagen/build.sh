#!/bin/sh
LIBMUTAGEN_SRC_DIR=$(dirname $0)
LIBFUZZER_SRC_DIR=$LIBMUTAGEN_SRC_DIR/..
CXX="${CXX:-clang}"
for f in $LIBMUTAGEN_SRC_DIR/*.cpp; do
  $CXX -g -O2 -fno-omit-frame-pointer -std=c++11 $f -c -I$LIBFUZZER_SRC_DIR &
done
wait
rm -f libMutagen.a
ar ru libMutagen.a Mutagen*.o
rm -f Mutagen*.o


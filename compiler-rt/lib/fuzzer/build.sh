#!/bin/sh
LIBFUZZER_SRC_DIR=$(dirname $0)
LIBMUTAGEN_SRC_DIR=$LIBFUZZER_SRC_DIR/mutagen
CXX="${CXX:-clang}"
for f in $LIBFUZZER_SRC_DIR/*.cpp $LIBMUTAGEN_SRC_DIR/*.cpp; do
  $CXX -g -O2 -fno-omit-frame-pointer -std=c++11 $f -c -I$LIBFUZZER_SRC_DIR &
done
wait
rm -f libFuzzer.a
ar ru libFuzzer.a Fuzzer*.o Mutagen*.o
rm -f Fuzzer*.o Mutagen*.o

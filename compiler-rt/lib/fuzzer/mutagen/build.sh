#!/bin/sh
set -e
LIBMUTAGEN_SRC_DIR=$(dirname $0)
CXX="${CXX:-clang}"
for f in $LIBMUTAGEN_SRC_DIR/*.cpp; do
  $CXX -O2 -fno-omit-frame-pointer -std=c++11 -I$LIBMUTAGEN_SRC_DIR/.. $f -c &
done
wait
rm -f libMutagen.a
ar ru libMutagen.a Mutagen*.o
rm -f Mutagen*.o


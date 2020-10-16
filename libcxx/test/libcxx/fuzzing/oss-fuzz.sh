#!/bin/bash -eu

#
# This script runs the continuous fuzzing tests on OSS-Fuzz.
#

if [[ $SANITIZER = *undefined* ]]; then
  CXXFLAGS="$CXXFLAGS -fsanitize=unsigned-integer-overflow -fsanitize-trap=unsigned-integer-overflow"
fi

for f in $(grep -v "#" libcxx/fuzzing/RoutineNames.txt); do
  cat > ${f}_fuzzer.cc <<EOF
#include "fuzzing/fuzzing.h"
#include <cassert>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  int result = fuzzing::$f(data, size);
  assert(result == 0); return 0;
}
EOF
  $CXX $CXXFLAGS -std=c++11 ${f}_fuzzer.cc ./libcxx/fuzzing/fuzzing.cpp \
      -nostdinc++ -cxx-isystem ./libcxx/include -iquote ./libcxx \
      -o $OUT/$f $LIB_FUZZING_ENGINE
done

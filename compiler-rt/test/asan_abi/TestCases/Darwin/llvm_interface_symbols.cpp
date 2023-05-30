// RUN: %clang_asan_abi -O0 -c -fsanitize-stable-abi -fsanitize=address %s -o %t.o
// RUN: %clangxx -c %p/../../../../lib/asan_abi/asan_abi.cpp -o asan_abi.o
// RUN: %clangxx -dead_strip -o %t %t.o %libasan_abi asan_abi.o && %run %t 2>&1
// RUN: %clangxx -x c++-header -o - -E %p/../../../../lib/asan/asan_interface.inc \
// RUN: | sed "s/INTERFACE_FUNCTION/\nINTERFACE_FUNCTION/g" > %t.asan_interface.inc
// RUN: llvm-nm -g %libasan_abi                                   \
// RUN: | grep " [TU] "                                           \
// RUN: | grep -o "\(__asan\)[^ ]*"                               \
// RUN: | grep -v "\(__asan_abi\)[^ ]*"                           \
// RUN: | sed -e "s/__asan_version_mismatch_check_v[0-9]+/__asan_version_mismatch_check/" \
// RUN: > %t.exports
// RUN: sed -e ':a' -e 'N' -e '$!ba'                              \
// RUN:     -e 's/ //g'                                           \
// RUN:     -e ':b' -e 's/\n\n/\n/g' -e 'tb'                      \
// RUN:     -e 's/(\n/(/g'                                        \
// RUN: %t.asan_interface.inc                                     \
// RUN: | grep -v -f %p/../../../../lib/asan_abi/asan_abi_tbd.txt \
// RUN: | grep -e "INTERFACE_\(WEAK_\)\?FUNCTION"                 \
// RUN: | grep -v "__sanitizer[^ ]*"                              \
// RUN: | sed -e "s/.*(//" -e "s/).*//" > %t.imports
// RUN: sort %t.imports | uniq > %t.imports-sorted
// RUN: sort %t.exports | uniq > %t.exports-sorted
// RUN: diff %t.imports-sorted %t.exports-sorted

// UNSUPPORTED: ios

int main() { return 0; }

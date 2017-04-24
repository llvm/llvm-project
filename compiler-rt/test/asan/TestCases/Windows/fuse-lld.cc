// If we have LLD, see that things more or less work.
//
// REQUIRES: lld-available
//
// FIXME: Use -fuse-ld=lld after the old COFF linker is removed.
// FIXME: Test will fail until we add flags for requesting dwarf or cv.
// RUNX: %clangxx_asan -O2 %s -o %t.exe -fuse-ld=lld -Wl,-debug
// RUN: %clangxx_asan -c -O2 %s -o %t.o -g -gdwarf
// RUN: lld-link %t.o -out:%t.exe -debug -nopdb -defaultlib:libcmt %asan_lib %asan_cxx_lib
// RUN: not %run %t.exe 2>&1 | FileCheck %s

#include <stdlib.h>

int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: heap-use-after-free
  // CHECK: free
  // CHECK: main{{.*}}fuse-lld.cc:[[@LINE-4]]:3
  // CHECK: malloc
  // CHECK: main{{.*}}fuse-lld.cc:[[@LINE-7]]:20
}

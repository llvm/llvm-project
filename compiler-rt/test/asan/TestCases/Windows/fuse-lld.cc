// If we have LLD, see that things more or less work.
//
// REQUIRES: lld-available
//
// RUN: %clangxx_asan -O2 %s -o %t.exe -g -gcodeview -fuse-ld=lld -Wl,-debug
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

// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=print_stats=1 not %run %t 2>&1 | FileCheck %s

// FIXME: Fix this test under GCC.
// REQUIRES: Clang

#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char*)malloc(10 * sizeof(char));
  memset(x, 0, 10);
  int res = x[argc * 10];  // BOOOM
  // CHECK: {{READ of size 1 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in main .*heap-overflow.cpp:}}[[@LINE-2]]
  // CHECK: {{0x.* is located 0 bytes to the right of 10-byte region}}
  // CHECK: {{allocated by thread T0 here:}}

  // CHECK: {{    #0 0x.* in .*malloc}}
  free(x);
  return res;
}

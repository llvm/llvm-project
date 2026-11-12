// RUN: %clangxx_msan -O0 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -O1 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -O2 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -O3 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <stdlib.h>
int main(int argc, char **argv) {
  int x;
  int *volatile p = &x;
  int z = *p ? 1 : 0;
  if (z)
    exit(0);
  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: {{#0 0x.* in main .*select.cpp:}}[[@LINE-3]]

  // CHECK: SUMMARY: MemorySanitizer: use-of-uninitialized-value {{.*select.cpp:.* main}}
  return 0;
}

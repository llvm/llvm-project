// Regression test for origin propagation in "select i1, float, float".
// https://code.google.com/p/memory-sanitizer/issues/detail?id=78

// RUN: %clangxx_msan -O2 -fsanitize-memory-track-origins %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_msan -O2 -fsanitize-memory-track-origins=2 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <stdio.h>
#include <sanitizer/msan_interface.h>

int main() {
  volatile bool b = true;
  float x, y;
  __msan_allocated_memory(&x, sizeof(x));
  __msan_allocated_memory(&y, sizeof(y));
  float z = b ? x : y;
  if (z > 0) printf(".\n");
  // CHECK: Memory was marked as uninitialized
  // CHECK: {{#0 0x.* in .*__msan_allocated_memory}}
  // CHECK: {{#1 0x.* in main .*select_float_origin.cpp:}}[[@LINE-6]]
  return 0;
}

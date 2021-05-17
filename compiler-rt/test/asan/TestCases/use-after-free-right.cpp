// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s
// REQUIRES: stable-runtime

// Test use-after-free report in the case when access is at the right border of
// the allocation.

#include <stdlib.h>
int main() {
  volatile char *x = (char*)malloc(sizeof(char));
  free((void*)x);
  *x = 42;
  // CHECK: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
  // CHECK:   {{0x.* at pc 0x.* bp 0x.* sp 0x.*}}
  // CHECK: {{WRITE of size 1 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in main .*use-after-free-right.cpp:}}[[@LINE-4]]
  // CHECK: {{0x.* is located 0 bytes inside of 1-byte region .0x.*,0x.*}}
  // CHECK: {{freed by thread T0 here:}}
  // CHECK: {{    #0 0x.* in .*free}}
  // CHECK: {{    #1 0x.* in main .*use-after-free-right.cpp:}}[[@LINE-9]]

  // CHECK: {{previously allocated by thread T0 here:}}
  // CHECK: {{    #0 0x.* in .*malloc}}
  // CHECK: {{    #1 0x.* in main .*use-after-free-right.cpp:}}[[@LINE-14]]
}

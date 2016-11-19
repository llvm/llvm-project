// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK

// REQUIRES: compiler-rt-optimized
// REQUIRES: stable-runtime

#include <string.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  char *hello = (char*)malloc(6);
  strcpy(hello, "hello");
  char *short_buffer = (char*)malloc(9);
  strncpy(short_buffer, hello, 10);  // BOOM
  // CHECK: {{WRITE of size 10 at 0x.* thread T0}}
  // CHECK-Linux: {{    #0 0x.* in .*strncpy}}
  // CHECK-Darwin: {{    #0 0x.* in wrap_strncpy}}
  // CHECK: {{    #1 0x.* in main .*strncpy-overflow.cc:}}[[@LINE-4]]
  // CHECK: {{0x.* is located 0 bytes to the right of 9-byte region}}
  // CHECK: {{allocated by thread T0 here:}}

  // CHECK-Linux: {{    #0 0x.* in .*malloc}}
  // CHECK-Linux: {{    #1 0x.* in main .*strncpy-overflow.cc:}}[[@LINE-10]]

  // CHECK-Darwin: {{    #0 0x.* in wrap_malloc.*}}
  // CHECK-Darwin: {{    #1 0x.* in main .*strncpy-overflow.cc:}}[[@LINE-13]]
  return short_buffer[8];
}

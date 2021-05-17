// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

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
  // CHECK: {{    #0 0x.* in .*strncpy}}
  // CHECK: {{    #1 0x.* in main .*strncpy-overflow.cpp:}}[[@LINE-3]]
  // CHECK: {{0x.* is located 0 bytes to the right of 9-byte region}}
  // CHECK: {{allocated by thread T0 here:}}
  // CHECK: {{    #0 0x.* in .*malloc}}
  // CHECK: {{    #1 0x.* in main .*strncpy-overflow.cpp:}}[[@LINE-8]]
  return short_buffer[8];
}

// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

// REQUIRES: compiler-rt-optimized
// REQUIRES: stable-runtime

#include <string.h>
#include <stdlib.h>

// We need a way to prevent the optimize from eliminating the
// strncpy below (which otherwises writes to dead storage).  We
// need the read to be out-of-line to prevent memory forwarding
// from making the memory dead again.
int sink_memory(int N, char *p) __attribute__((noinline));
int sink_memory(int N, char *p) {
  int sum = 0;
  for (int i = 0; i < N; i++)
    sum += p[i];
  return sum;
}

int main(int argc, char **argv) {
  char *hello = (char*)malloc(6);
  strcpy(hello, "hello");
  int rval = sink_memory(6, hello);
  char *short_buffer = (char*)malloc(9);
  strncpy(short_buffer, hello, 10);  // BOOM
  // CHECK: {{WRITE of size 10 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in .*strncpy}}
  // CHECK: {{    #1 0x.* in main .*strncpy-overflow.cpp:}}[[@LINE-3]]
  // CHECK: {{0x.* is located 0 bytes after 9-byte region}}
  // CHECK: {{allocated by thread T0 here:}}
  // CHECK: {{    #0 0x.* in .*malloc}}
  // CHECK: {{    #1 0x.* in main .*strncpy-overflow.cpp:}}[[@LINE-8]]
  return rval + sink_memory(9, short_buffer);
}

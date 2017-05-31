// RUN: %clangxx_asan -O0 -mllvm -asan-instrument-allocas %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
//

#include <assert.h>

__attribute__((noinline)) void foo(int index, int len) {
  volatile char str[len] __attribute__((aligned(128)));
  assert(!(reinterpret_cast<long>(str) & 127L));
  str[index] = '1'; // BOOM
// CHECK: ERROR: AddressSanitizer: dynamic-stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] thread T0
}

int main(int argc, char **argv) {
  foo(10, 10);
  return 0;
}

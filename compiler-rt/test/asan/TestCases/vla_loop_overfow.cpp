// RUN: %clangxx_asan -O0 -mllvm -asan-instrument-dynamic-allocas %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
//
// REQUIRES: stable-runtime

#include <assert.h>
#include <stdint.h>

void foo(int index, int len) {
  for (int i = 1; i < len; ++i) {
    char array[len]; // NOLINT
    assert(!(reinterpret_cast<uintptr_t>(array) & 31L));
    array[index + i] = 0;
// CHECK: ERROR: AddressSanitizer: dynamic-stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] thread T0
  }
}

int main(int argc, char **argv) {
  foo(9, 21);
  return 0;
}

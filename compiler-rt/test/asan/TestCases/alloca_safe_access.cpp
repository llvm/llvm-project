// RUN: %clangxx_asan -O0 -mllvm -asan-instrument-dynamic-allocas %s -o %t
// RUN: %run %t 2>&1
//

#include <assert.h>
#include <stdint.h>

__attribute__((noinline)) void foo(int index, int len) {
  volatile char str[len] __attribute__((aligned(32)));
  assert(!(reinterpret_cast<uintptr_t>(str) & 31L));
  str[index] = '1';
}

int main(int argc, char **argv) {
  foo(4, 5);
  foo(39, 40);
  return 0;
}

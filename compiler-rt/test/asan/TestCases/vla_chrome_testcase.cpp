// RUN: %clangxx_asan -O0 -mllvm -asan-instrument-dynamic-allocas %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
//

// This is reduced testcase based on Chromium code.
// See http://reviews.llvm.org/D6055?vs=on&id=15616&whitespace=ignore-all#toc.

#include <stdint.h>
#include <assert.h>

int a = 7;
int b;
int c;
int *p;

__attribute__((noinline)) void fn3(int *first, int second) {
}

int main() {
  int d = b && c;
  int e[a];
  assert(!(reinterpret_cast<uintptr_t>(e) & 31L));
  int f;
  if (d)
    fn3(&f, sizeof 0 * (&c - e));
  e[a] = 0;
// CHECK: ERROR: AddressSanitizer: dynamic-stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]] thread T0
  return 0;
}

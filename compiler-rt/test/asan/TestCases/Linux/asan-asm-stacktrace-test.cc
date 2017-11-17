// Check that a stack unwinding algorithm works corretly even with the assembly
// instrumentation.

// REQUIRES: x86_64-target-arch, shadow-scale-3
// RUN: %clangxx_asan -g -O1 %s -fno-inline-functions -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -mllvm -asan-instrument-assembly -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -g -O1 %s -fno-inline-functions -fomit-frame-pointer -momit-leaf-frame-pointer -mllvm -asan-instrument-assembly -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -g0 -O1 %s -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-exceptions -fno-inline-functions -fomit-frame-pointer -momit-leaf-frame-pointer -mllvm -asan-instrument-assembly -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-nounwind

#include <cstddef>

// CHECK: READ of size 4
// CHECK-NEXT: {{#0 0x[0-9a-fA-F]+ in foo}}
// CHECK-NEXT: {{#1 0x[0-9a-fA-F]+ in main}}

// CHECK-nounwind: READ of size 4
// CHECK-nounwind-NEXT: {{#0 0x[0-9a-fA-F]+ in foo}}

__attribute__((noinline)) int foo(size_t n, int *buffer) {
  int r;
  __asm__("movl (%[buffer], %[n], 4), %[r]  \n\t"
          : [r] "=r"(r)
          : [buffer] "r"(buffer), [n] "r"(n)
          : "memory");
  return r;
}

int main() {
  const size_t n = 16;
  int *buffer = new int[n];
  foo(n, buffer);
  delete[] buffer;
  return 0;
}

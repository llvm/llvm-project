// Test that use-after-return works with arguments passed by value.
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=detect_stack_use_after_return=0 %run %t 2>&1 | \
// RUN:     FileCheck --check-prefix=CHECK-NO-UAR %s
// RUN: not %env_asan_opts=detect_stack_use_after_return=1 %run %t 2>&1 | \
// RUN:     FileCheck --check-prefix=CHECK-UAR %s

#include <cstdio>

struct A {
  int a[8];
};

A *foo(A a) {
  return &a;
}

int main() {
  A *a = foo(A());
  a->a[0] = 7;
  std::fprintf(stderr, "\n");  // Ensures some output is generated for FileCheck
                               // to verify in the case where UAR is not
                               // detected.
}

// CHECK-NO-UAR-NOT: ERROR: AddressSanitizer: stack-use-after-return
// CHECK-NO-UAR-NOT: WRITE of size 4 at
// CHECK-NO-UAR-NOT: Memory access at offset {{[0-9]+}} is inside this variable
//
// CHECK-UAR: ERROR: AddressSanitizer: stack-use-after-return
// CHECK-UAR: WRITE of size 4 at
// CHECK-UAR: Memory access at offset {{[0-9]+}} is inside this variable

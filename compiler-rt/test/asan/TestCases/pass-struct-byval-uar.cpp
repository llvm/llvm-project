// Test that use-after-return works with arguments passed by value.
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=detect_stack_use_after_return=0 %run %t 2>&1 | \
// RUN:     FileCheck --check-prefix=CHECK-NO-UAR %s
// RUN: not %env_asan_opts=detect_stack_use_after_return=1 %run %t 2>&1 | \
// RUN:     FileCheck --check-prefix=CHECK-UAR %s
// RUN: %clangxx_asan -O0 %s -o %t -fsanitize-address-use-after-return=never && \
// RUN:     %run %t 2>&1 | FileCheck --check-prefix=CHECK-NO-UAR %s
// RUN: %clangxx_asan -O0 %s -o %t -fsanitize-address-use-after-return=always && \
// RUN:     not %run %t 2>&1 | FileCheck --check-prefix=CHECK-UAR %s
//
// On several architectures, the IR does not use byval arguments for foo() and
// instead creates a copy in main() and gives foo() a pointer to the copy.  In
// that case, ASAN has nothing to poison on return from foo() and will not
// detect the UAR.
// REQUIRES: x86_64-target-arch, linux, !android

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

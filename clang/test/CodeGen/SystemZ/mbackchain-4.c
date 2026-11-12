// REQUIRES: systemz-registered-target
// RUN: %clang --target=s390x-linux -O1 -S -o - %s | FileCheck %s

__attribute__((target("backchain")))
void *foo(void) {
  return __builtin_return_address(1);
}

// CHECK-LABEL: foo:
// CHECK: lg %r1, 0(%r15)
// CHECK: lg %r2, 112(%r1)
// CHECK: br %r14

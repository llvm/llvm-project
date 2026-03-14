// RUN: not %clang_cc1 -triple s390x-linux-gnu -O2 -emit-llvm -o - %s 2>&1 \
// RUN:  | FileCheck %s
// REQUIRES: systemz-registered-target

// Test that an error is given if a physreg is defined by multiple operands.
int test_physreg_defs(void) {
  register int l __asm__("r7") = 0;

  // CHECK: error: multiple outputs to hard register: r7
  __asm__("" : "+r"(l), "=r"(l));

  return l;
}

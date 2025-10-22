
// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +zicboz -O2 -S -o - %s | FileCheck %s

#include <riscv_cmo.h>

void test(void *x) {
// CHECK-LABEL: test:
// CHECK:       cbo.zero (a0)
// CHECK:       ret
  __riscv_cbo_zero(x);
}


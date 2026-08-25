// REQUIRES: riscv-registered-target
// expected-no-diagnostics

// RUN: %clang %s -O2 -S -o - --target=riscv32 -march=rv32i_xcvelw \
// RUN:   -Werror -Wextra -Xclang -verify \
// RUN:   | FileCheck %s

#include <riscv_corev_elw.h>

// CHECK-LABEL: test_elw_elw:
// CHECK: cv.elw
uint32_t test_elw_elw(void *a0) { return __riscv_cv_elw_elw(a0); }

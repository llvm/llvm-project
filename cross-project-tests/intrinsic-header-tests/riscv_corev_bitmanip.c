// REQUIRES: riscv-registered-target
// expected-no-diagnostics

// RUN: %clang %s -O2 -S -o - --target=riscv32 -march=rv32i_xcvbitmanip \
// RUN:   -Werror -Wextra -Xclang -verify \
// RUN:   | FileCheck %s

#include <riscv_corev_bitmanip.h>

// The extract/extractu/bclr/bset/insert ops have two forms:
// - register-operand form (cv.*r, position from a register)
// - immediate form (cv.*, position as a constant)
// Both are exercised below (*_r vs *_i tests).

// CHECK-LABEL: test_extract_r:
// CHECK: cv.extractr
int32_t test_extract_r(uint32_t a0, uint32_t a1) {
  return __riscv_cv_bitmanip_extract(a0, a1);
}

// CHECK-LABEL: test_extract_i:
// CHECK: cv.extract {{.*}}, 5
int32_t test_extract_i(uint32_t a0) {
  return __riscv_cv_bitmanip_extract(a0, 5);
}

// CHECK-LABEL: test_extractu_r:
// CHECK: cv.extractur
uint32_t test_extractu_r(uint32_t a0, uint32_t a1) {
  return __riscv_cv_bitmanip_extractu(a0, a1);
}

// CHECK-LABEL: test_extractu_i:
// CHECK: cv.extractu {{.*}}, 5
uint32_t test_extractu_i(uint32_t a0) {
  return __riscv_cv_bitmanip_extractu(a0, 5);
}

// CHECK-LABEL: test_bclr_r:
// CHECK: cv.bclrr
uint32_t test_bclr_r(uint32_t a0, uint32_t a1) {
  return __riscv_cv_bitmanip_bclr(a0, a1);
}

// CHECK-LABEL: test_bclr_i:
// CHECK: cv.bclr {{.*}}, 5
uint32_t test_bclr_i(uint32_t a0) { return __riscv_cv_bitmanip_bclr(a0, 5); }

// CHECK-LABEL: test_bset_r:
// CHECK: cv.bsetr
uint32_t test_bset_r(uint32_t a0, uint32_t a1) {
  return __riscv_cv_bitmanip_bset(a0, a1);
}

// CHECK-LABEL: test_bset_i:
// CHECK: cv.bset {{.*}}, 5
uint32_t test_bset_i(uint32_t a0) { return __riscv_cv_bitmanip_bset(a0, 5); }

// CHECK-LABEL: test_insert_r:
// CHECK: cv.insertr
uint32_t test_insert_r(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_bitmanip_insert(a0, a1, a2);
}

// CHECK-LABEL: test_insert_i:
// CHECK: cv.insert {{.*}}, 5
uint32_t test_insert_i(uint32_t a0, uint32_t a1) {
  return __riscv_cv_bitmanip_insert(a0, a1, 5);
}

// CHECK-LABEL: test_clb:
// CHECK: cv.clb
uint32_t test_clb(uint32_t a0) { return __riscv_cv_bitmanip_clb(a0); }

// CHECK-LABEL: test_bitrev:
// CHECK: cv.bitrev
uint32_t test_bitrev(uint32_t a0) {
  return __riscv_cv_bitmanip_bitrev(a0, 1, 1);
}

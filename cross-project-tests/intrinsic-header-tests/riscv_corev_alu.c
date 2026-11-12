// REQUIRES: riscv-registered-target
// expected-no-diagnostics

// RUN: %clang %s -O2 -S -o - --target=riscv32 -march=rv32i_xcvalu \
// RUN:   -Werror -Wextra -Xclang -verify \
// RUN:   | FileCheck %s

#include <riscv_corev_alu.h>
#include <stdint.h>

// The exths/exthz/extbs/extbz builtins fold to a no-op for an already
// register-width argument, so they emit no cv.* instruction and are
// covered by the IR-level test instead; they are omitted here.

// CHECK-LABEL: test_alu_abs:
// CHECK: cv.abs
long test_alu_abs(long a) { return __riscv_cv_abs(a); }

// CHECK-LABEL: test_alu_sle:
// CHECK: cv.sle {{.*}}
long test_alu_sle(long a, long b) { return __riscv_cv_alu_sle(a, b); }

// CHECK-LABEL: test_alu_sleu:
// CHECK: cv.sleu
long test_alu_sleu(unsigned long a, unsigned long b) {
  return __riscv_cv_alu_sleu(a, b);
}

// CHECK-LABEL: test_alu_min:
// CHECK: cv.min {{.*}}
long test_alu_min(long a, long b) { return __riscv_cv_alu_min(a, b); }

// CHECK-LABEL: test_alu_minu:
// CHECK: cv.minu
unsigned long test_alu_minu(unsigned long a, unsigned long b) {
  return __riscv_cv_alu_minu(a, b);
}

// CHECK-LABEL: test_alu_max:
// CHECK: cv.max {{.*}}
long test_alu_max(long a, long b) { return __riscv_cv_alu_max(a, b); }

// CHECK-LABEL: test_alu_maxu:
// CHECK: cv.maxu
unsigned long test_alu_maxu(unsigned long a, unsigned long b) {
  return __riscv_cv_alu_maxu(a, b);
}

// CHECK-LABEL: test_alu_clip:
// CHECK: cv.clip {{.*}}, 4
long test_alu_clip(long a) { return __riscv_cv_alu_clip(a, 7); }

// CHECK-LABEL: test_alu_clipu:
// CHECK: cv.clipu {{.*}}, 4
unsigned long test_alu_clipu(unsigned long a) {
  return __riscv_cv_alu_clipu(a, 7);
}

// CHECK-LABEL: test_alu_addN:
// CHECK: cv.addn {{.*}}, 3
long test_alu_addN(long a, long b) { return __riscv_cv_alu_addN(a, b, 3); }

// CHECK-LABEL: test_alu_adduN:
// CHECK: cv.addun {{.*}}, 3
unsigned long test_alu_adduN(unsigned long a, unsigned long b) {
  return __riscv_cv_alu_adduN(a, b, 3);
}

// CHECK-LABEL: test_alu_addRN:
// CHECK: cv.addrn {{.*}}, 3
long test_alu_addRN(long a, long b) { return __riscv_cv_alu_addRN(a, b, 3); }

// CHECK-LABEL: test_alu_adduRN:
// CHECK: cv.addurn {{.*}}, 3
unsigned long test_alu_adduRN(unsigned long a, unsigned long b) {
  return __riscv_cv_alu_adduRN(a, b, 3);
}

// CHECK-LABEL: test_alu_subN:
// CHECK: cv.subn {{.*}}, 3
long test_alu_subN(long a, long b) { return __riscv_cv_alu_subN(a, b, 3); }

// CHECK-LABEL: test_alu_subuN:
// CHECK: cv.subun {{.*}}, 3
unsigned long test_alu_subuN(unsigned long a, unsigned long b) {
  return __riscv_cv_alu_subuN(a, b, 3);
}

// CHECK-LABEL: test_alu_subRN:
// CHECK: cv.subrn {{.*}}, 3
long test_alu_subRN(long a, long b) { return __riscv_cv_alu_subRN(a, b, 3); }

// CHECK-LABEL: test_alu_subuRN:
// CHECK: cv.suburn {{.*}}, 3
unsigned long test_alu_subuRN(unsigned long a, unsigned long b) {
  return __riscv_cv_alu_subuRN(a, b, 3);
}

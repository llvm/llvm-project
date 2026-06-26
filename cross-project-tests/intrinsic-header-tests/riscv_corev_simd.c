// REQUIRES: riscv-registered-target
// expected-no-diagnostics

// RUN: %clang %s -O2 -S -o - --target=riscv32 -march=rv32i_xcvsimd \
// RUN:   -Werror -Wextra -Xclang -verify \
// RUN:   | FileCheck %s

#include <riscv_corev_simd.h>

// CHECK-LABEL: test_simd_add_h:
// CHECK: cv.add.h
uint32_t test_simd_add_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_add_h(a0, a1);
}

// CHECK-LABEL: test_simd_add_b:
// CHECK: cv.add.b
uint32_t test_simd_add_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_add_b(a0, a1);
}

// CHECK-LABEL: test_simd_sub_h:
// CHECK: cv.sub.h
uint32_t test_simd_sub_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_sub_h(a0, a1);
}

// CHECK-LABEL: test_simd_sub_b:
// CHECK: cv.sub.b
uint32_t test_simd_sub_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_sub_b(a0, a1);
}

// CHECK-LABEL: test_simd_min_h:
// CHECK: cv.min.h
uint32_t test_simd_min_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_min_h(a0, a1);
}

// CHECK-LABEL: test_simd_min_b:
// CHECK: cv.min.b
uint32_t test_simd_min_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_min_b(a0, a1);
}

// CHECK-LABEL: test_simd_minu_h:
// CHECK: cv.minu.h
uint32_t test_simd_minu_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_minu_h(a0, a1);
}

// CHECK-LABEL: test_simd_minu_b:
// CHECK: cv.minu.b
uint32_t test_simd_minu_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_minu_b(a0, a1);
}

// CHECK-LABEL: test_simd_max_h:
// CHECK: cv.max.h
uint32_t test_simd_max_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_max_h(a0, a1);
}

// CHECK-LABEL: test_simd_max_b:
// CHECK: cv.max.b
uint32_t test_simd_max_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_max_b(a0, a1);
}

// CHECK-LABEL: test_simd_maxu_h:
// CHECK: cv.maxu.h
uint32_t test_simd_maxu_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_maxu_h(a0, a1);
}

// CHECK-LABEL: test_simd_maxu_b:
// CHECK: cv.maxu.b
uint32_t test_simd_maxu_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_maxu_b(a0, a1);
}

// CHECK-LABEL: test_simd_and_h:
// CHECK: and a0,{{.*}}
uint32_t test_simd_and_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_and_h(a0, a1);
}

// CHECK-LABEL: test_simd_and_b:
// CHECK: and a0,{{.*}}
uint32_t test_simd_and_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_and_b(a0, a1);
}

// CHECK-LABEL: test_simd_or_h:
// CHECK: or a0,{{.*}}
uint32_t test_simd_or_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_or_h(a0, a1);
}

// CHECK-LABEL: test_simd_or_b:
// CHECK: or a0,{{.*}}
uint32_t test_simd_or_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_or_b(a0, a1);
}

// CHECK-LABEL: test_simd_xor_h:
// CHECK: xor a0,{{.*}}
uint32_t test_simd_xor_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_xor_h(a0, a1);
}

// CHECK-LABEL: test_simd_xor_b:
// CHECK: xor a0,{{.*}}
uint32_t test_simd_xor_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_xor_b(a0, a1);
}

// CHECK-LABEL: test_simd_abs_h:
// CHECK: cv.abs.h
uint32_t test_simd_abs_h(uint32_t a0) { return __riscv_cv_simd_abs_h(a0); }

// CHECK-LABEL: test_simd_abs_b:
// CHECK: cv.abs.b
uint32_t test_simd_abs_b(uint32_t a0) { return __riscv_cv_simd_abs_b(a0); }

// CHECK-LABEL: test_simd_dotup_h:
// CHECK: cv.dotup.h
uint32_t test_simd_dotup_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_dotup_h(a0, a1);
}

// CHECK-LABEL: test_simd_dotup_b:
// CHECK: cv.dotup.b
uint32_t test_simd_dotup_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_dotup_b(a0, a1);
}

// CHECK-LABEL: test_simd_dotup_sc_h:
// CHECK: cv.dotup.sc.h
uint32_t test_simd_dotup_sc_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_dotup_sc_h(a0, a1);
}

// CHECK-LABEL: test_simd_dotup_sc_b:
// CHECK: cv.dotup.sc.b
uint32_t test_simd_dotup_sc_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_dotup_sc_b(a0, a1);
}

// CHECK-LABEL: test_simd_dotusp_h:
// CHECK: cv.dotusp.h
int32_t test_simd_dotusp_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_dotusp_h(a0, a1);
}

// CHECK-LABEL: test_simd_dotusp_b:
// CHECK: cv.dotusp.b
int32_t test_simd_dotusp_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_dotusp_b(a0, a1);
}

// CHECK-LABEL: test_simd_dotusp_sc_h:
// CHECK: cv.dotusp.sc.h
int32_t test_simd_dotusp_sc_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_dotusp_sc_h(a0, a1);
}

// CHECK-LABEL: test_simd_dotusp_sc_b:
// CHECK: cv.dotusp.sc.b
int32_t test_simd_dotusp_sc_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_dotusp_sc_b(a0, a1);
}

// CHECK-LABEL: test_simd_dotsp_h:
// CHECK: cv.dotsp.h
int32_t test_simd_dotsp_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_dotsp_h(a0, a1);
}

// CHECK-LABEL: test_simd_dotsp_b:
// CHECK: cv.dotsp.b
int32_t test_simd_dotsp_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_dotsp_b(a0, a1);
}

// CHECK-LABEL: test_simd_dotsp_sc_h:
// CHECK: cv.dotsp.sc.h
int32_t test_simd_dotsp_sc_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_dotsp_sc_h(a0, a1);
}

// CHECK-LABEL: test_simd_dotsp_sc_b:
// CHECK: cv.dotsp.sc.b
int32_t test_simd_dotsp_sc_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_dotsp_sc_b(a0, a1);
}

// CHECK-LABEL: test_simd_sdotup_h:
// CHECK: cv.sdotup.h
uint32_t test_simd_sdotup_h(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_simd_sdotup_h(a0, a1, a2);
}

// CHECK-LABEL: test_simd_sdotup_b:
// CHECK: cv.sdotup.b
uint32_t test_simd_sdotup_b(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_simd_sdotup_b(a0, a1, a2);
}

// CHECK-LABEL: test_simd_sdotup_sc_h:
// CHECK: cv.sdotup.sc.h
uint32_t test_simd_sdotup_sc_h(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_simd_sdotup_sc_h(a0, a1, a2);
}

// CHECK-LABEL: test_simd_sdotup_sc_b:
// CHECK: cv.sdotup.sc.b
uint32_t test_simd_sdotup_sc_b(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_simd_sdotup_sc_b(a0, a1, a2);
}

// CHECK-LABEL: test_simd_sdotusp_h:
// CHECK: cv.sdotusp.h
int32_t test_simd_sdotusp_h(uint32_t a0, uint32_t a1, int32_t a2) {
  return __riscv_cv_simd_sdotusp_h(a0, a1, a2);
}

// CHECK-LABEL: test_simd_sdotusp_b:
// CHECK: cv.sdotusp.b
int32_t test_simd_sdotusp_b(uint32_t a0, uint32_t a1, int32_t a2) {
  return __riscv_cv_simd_sdotusp_b(a0, a1, a2);
}

// CHECK-LABEL: test_simd_sdotusp_sc_h:
// CHECK: cv.sdotusp.sc.h
int32_t test_simd_sdotusp_sc_h(uint32_t a0, uint32_t a1, int32_t a2) {
  return __riscv_cv_simd_sdotusp_sc_h(a0, a1, a2);
}

// CHECK-LABEL: test_simd_sdotusp_sc_b:
// CHECK: cv.sdotusp.sc.b
int32_t test_simd_sdotusp_sc_b(uint32_t a0, uint32_t a1, int32_t a2) {
  return __riscv_cv_simd_sdotusp_sc_b(a0, a1, a2);
}

// CHECK-LABEL: test_simd_sdotsp_h:
// CHECK: cv.sdotsp.h
int32_t test_simd_sdotsp_h(uint32_t a0, uint32_t a1, int32_t a2) {
  return __riscv_cv_simd_sdotsp_h(a0, a1, a2);
}

// CHECK-LABEL: test_simd_sdotsp_b:
// CHECK: cv.sdotsp.b
int32_t test_simd_sdotsp_b(uint32_t a0, uint32_t a1, int32_t a2) {
  return __riscv_cv_simd_sdotsp_b(a0, a1, a2);
}

// CHECK-LABEL: test_simd_sdotsp_sc_h:
// CHECK: cv.sdotsp.sc.h
int32_t test_simd_sdotsp_sc_h(uint32_t a0, uint32_t a1, int32_t a2) {
  return __riscv_cv_simd_sdotsp_sc_h(a0, a1, a2);
}

// CHECK-LABEL: test_simd_sdotsp_sc_b:
// CHECK: cv.sdotsp.sc.b
int32_t test_simd_sdotsp_sc_b(uint32_t a0, uint32_t a1, int32_t a2) {
  return __riscv_cv_simd_sdotsp_sc_b(a0, a1, a2);
}

// CHECK-LABEL: test_simd_extract_h:
// CHECK: cv.extract.h {{.*}}, 1
int32_t test_simd_extract_h(uint32_t a0) {
  return __riscv_cv_simd_extract_h(a0, 1);
}

// CHECK-LABEL: test_simd_extract_b:
// CHECK: cv.extract.b {{.*}}, 1
int32_t test_simd_extract_b(uint32_t a0) {
  return __riscv_cv_simd_extract_b(a0, 1);
}

// CHECK-LABEL: test_simd_extractu_h:
// CHECK: cv.extractu.h {{.*}}, 1
uint32_t test_simd_extractu_h(uint32_t a0) {
  return __riscv_cv_simd_extractu_h(a0, 1);
}

// CHECK-LABEL: test_simd_extractu_b:
// CHECK: cv.extractu.b {{.*}}, 1
uint32_t test_simd_extractu_b(uint32_t a0) {
  return __riscv_cv_simd_extractu_b(a0, 1);
}

// CHECK-LABEL: test_simd_insert_h:
// CHECK: cv.insert.h {{.*}}, 1
uint32_t test_simd_insert_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_insert_h(a0, a1, 1);
}

// CHECK-LABEL: test_simd_insert_b:
// CHECK: cv.insert.b {{.*}}, 1
uint32_t test_simd_insert_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_insert_b(a0, a1, 1);
}

// CHECK-LABEL: test_simd_shuffle_h:
// CHECK: cv.shuffle.h
uint32_t test_simd_shuffle_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_shuffle_h(a0, a1);
}

// CHECK-LABEL: test_simd_shuffle_b:
// CHECK: cv.shuffle.b
uint32_t test_simd_shuffle_b(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_shuffle_b(a0, a1);
}

// CHECK-LABEL: test_simd_shuffle_sci_h:
// CHECK: cv.shuffle.sci.h {{.*}}, 1
uint32_t test_simd_shuffle_sci_h(uint32_t a0) {
  return __riscv_cv_simd_shuffle_sci_h(a0, 1);
}

// CHECK-LABEL: test_simd_shuffle_sci_b:
// CHECK: cv.shufflei0.sci.b {{.*}}, 1
uint32_t test_simd_shuffle_sci_b(uint32_t a0) {
  return __riscv_cv_simd_shuffle_sci_b(a0, 1);
}

// CHECK-LABEL: test_simd_shuffle2_h:
// CHECK: cv.shuffle2.h
uint32_t test_simd_shuffle2_h(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_simd_shuffle2_h(a0, a1, a2);
}

// CHECK-LABEL: test_simd_shuffle2_b:
// CHECK: cv.shuffle2.b
uint32_t test_simd_shuffle2_b(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_simd_shuffle2_b(a0, a1, a2);
}

// CHECK-LABEL: test_simd_packhi_h:
// CHECK: cv.pack.h
uint32_t test_simd_packhi_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_packhi_h(a0, a1);
}

// CHECK-LABEL: test_simd_packlo_h:
// CHECK: cv.pack {{.*}}
uint32_t test_simd_packlo_h(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_packlo_h(a0, a1);
}

// CHECK-LABEL: test_simd_packhi_b:
// CHECK: cv.packhi.b
uint32_t test_simd_packhi_b(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_simd_packhi_b(a0, a1, a2);
}

// CHECK-LABEL: test_simd_packlo_b:
// CHECK: cv.packlo.b
uint32_t test_simd_packlo_b(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_simd_packlo_b(a0, a1, a2);
}

// CHECK-LABEL: test_simd_cplxmul_r:
// CHECK: cv.cplxmul.r
uint32_t test_simd_cplxmul_r(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_simd_cplxmul_r(a0, a1, a2, 0);
}

// CHECK-LABEL: test_simd_cplxmul_i:
// CHECK: cv.cplxmul.i
uint32_t test_simd_cplxmul_i(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_simd_cplxmul_i(a0, a1, a2, 0);
}

// CHECK-LABEL: test_simd_cplxconj:
// CHECK: cv.cplxconj
uint32_t test_simd_cplxconj(uint32_t a0) {
  return __riscv_cv_simd_cplxconj(a0);
}

// CHECK-LABEL: test_simd_subrotmj:
// CHECK: cv.subrotmj
uint32_t test_simd_subrotmj(uint32_t a0, uint32_t a1) {
  return __riscv_cv_simd_subrotmj(a0, a1, 0);
}

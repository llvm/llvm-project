// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +xsfvcp \
// RUN:     -target-feature +xsfvqmaccdod -target-feature +xsfvqmaccqoq \
// RUN:     -target-feature +experimental-zvfbfmin -target-feature +xsfvfwmaccqqq \
// RUN:     -target-feature +xsfvfnrclipxfqf %s -fsyntax-only -verify

// expected-no-diagnostics

#include <riscv_vector.h>
#include <sifive_vector.h>

vint8m1_t test_vloxei64_v_i8m1(const int8_t *base, vuint64m8_t bindex, size_t vl) {
  return __riscv_vloxei64(base, bindex, vl);
}

void test_vsoxei64_v_i8m1(int8_t *base, vuint64m8_t bindex, vint8m1_t value, size_t vl) {
  __riscv_vsoxei64(base, bindex, value, vl);
}

void test_sf_vc_x_se_u64m1(uint64_t rs1, size_t vl) {
  __riscv_sf_vc_x_se_u64m1(1, 1, 1, rs1, vl);
}

void test_xsfvqmaccdod(vint32m1_t vd, vint8m1_t vs1, vint8m1_t vs2, size_t vl) {
  __riscv_sf_vqmacc_2x8x2(vd, vs1, vs2, vl);
}

void test_xsfvqmaccqoq(vint32m1_t vd, vint8m1_t vs1, vint8m1_t vs2, size_t vl) {
  __riscv_sf_vqmacc_4x8x4(vd, vs1, vs2, vl);
}

void test_xsfvfwmaccqqq(vfloat32m1_t vd, vbfloat16m1_t vs1, vbfloat16mf2_t vs2, size_t vl) {
  __riscv_sf_vfwmacc_4x4x4(vd, vs1, vs2, vl);
}

void test_xsfvfnrclipxufqf(vfloat32m1_t vs1, float rs2, size_t vl) {
  __riscv_sf_vfnrclip_xu_f_qf(vs1, rs2, vl);
}

void test_xsfvfnrclipxfqf(vfloat32m1_t vs1, float rs2, size_t vl) {
  __riscv_sf_vfnrclip_x_f_qf(vs1, rs2, vl);
}

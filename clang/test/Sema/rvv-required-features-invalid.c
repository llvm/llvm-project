// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv32 -target-feature +v %s -fsyntax-only -verify

#include <riscv_vector.h>
#include <sifive_vector.h>

vint8m1_t test_vloxei64_v_i8m1(const int8_t *base, vuint64m8_t bindex, size_t vl) {
  return __riscv_vloxei64(base, bindex, vl); // expected-error {{builtin requires at least one of the following extensions: 64bit}}
}

void test_vsoxei64_v_i8m1(int8_t *base, vuint64m8_t bindex, vint8m1_t value, size_t vl) {
  __riscv_vsoxei64(base, bindex, value, vl); // expected-error {{builtin requires at least one of the following extensions: 64bit}}
}

void test_xsfvcp_sf_vc_x_se_u64m1(uint64_t rs1, size_t vl) {
  __riscv_sf_vc_x_se_u64m1(1, 1, 1, rs1, vl); // expected-error {{call to undeclared function '__riscv_sf_vc_x_se_u64m1'}}
}

void test_xsfvqmaccdod(vint32m8_t vd, vint8m1_t vs1, vint8m8_t vs2, size_t vl) {
  __riscv_sf_vqmacc_2x8x2(vd, vs1, vs2, vl); // expected-error {{builtin requires at least one of the following extensions: xsfvqmaccdod}}
}

void test_xsfvqmaccqoq(vint32m1_t vd, vint8m1_t vs1, vint8mf2_t vs2, size_t vl) {
  __riscv_sf_vqmacc_4x8x4(vd, vs1, vs2, vl); // expected-error {{builtin requires at least one of the following extensions: xsfvqmaccqoq}}
}

void test_xsfvfwmaccqqq(vfloat32m4_t vd, vbfloat16m1_t vs1, vbfloat16m2_t vs2, size_t vl) {
  // expected-error@-1 {{RISC-V type 'vbfloat16m1_t' (aka '__rvv_bfloat16m1_t') requires the 'zvfbfmin' extension}}
  // expected-error@-2 {{RISC-V type 'vbfloat16m2_t' (aka '__rvv_bfloat16m2_t') requires the 'zvfbfmin' extension}}
  __riscv_sf_vfwmacc_4x4x4(vd, vs1, vs2, vl); // expected-error {{RISC-V type 'vbfloat16m1_t' (aka '__rvv_bfloat16m1_t') requires the 'zvfbfmin' extension}}
  // expected-error@-1 {{RISC-V type 'vbfloat16m2_t' (aka '__rvv_bfloat16m2_t') requires the 'zvfbfmin' extension}}
  // expected-error@-2 {{RISC-V type '__rvv_bfloat16m1_t' requires the 'zvfbfmin' extension}}
  // expected-error@-3 {{RISC-V type '__rvv_bfloat16m2_t' requires the 'zvfbfmin' extension}}
  // expected-error@-4 {{builtin requires at least one of the following extensions: xsfvfwmaccqqq}}
}

void test_xsfvfnrclipxfqf(vfloat32m1_t vs2, float rs1, size_t vl) {
  __riscv_sf_vfnrclip_x_f_qf(vs2, rs1, vl); // expected-error {{builtin requires at least one of the following extensions: xsfvfnrclipxfqf}}
}

void test_xsfvfnrclipxufqf(vfloat32mf2_t vs2, float rs1, size_t vl) {
  __riscv_sf_vfnrclip_xu_f_qf(vs2, rs1, 2, vl); // expected-error {{builtin requires at least one of the following extensions: xsfvfnrclipxfqf}}
}

void test_zvfbfwma_vfwmaccbf16(vfloat32m4_t vd, __bf16 vs1, vbfloat16m2_t vs2, size_t vl) {
  // expected-error@-1 {{RISC-V type 'vbfloat16m2_t' (aka '__rvv_bfloat16m2_t') requires the 'zvfbfmin' extension}}
  __riscv_vfwmaccbf16(vd, vs1, vs2, vl); // expected-error {{RISC-V type 'vbfloat16m2_t' (aka '__rvv_bfloat16m2_t') requires the 'zvfbfmin' extension}}
  // expected-error@-1 {{RISC-V type '__rvv_bfloat16m2_t' requires the 'zvfbfmin' extension}}
  // expected-error@-2 {{builtin requires at least one of the following extensions: zvfbfwma}}
}

void test_zvfbfmin_vfwcvtbf16(vbfloat16m2_t vs2, size_t vl) {
  // expected-error@-1 {{RISC-V type 'vbfloat16m2_t' (aka '__rvv_bfloat16m2_t') requires the 'zvfbfmin' extension}}
  __riscv_vfwcvtbf16_f_f_v_f32m4(vs2, vl);; // expected-error {{RISC-V type 'vbfloat16m2_t' (aka '__rvv_bfloat16m2_t') requires the 'zvfbfmin' extension}}
  // expected-error@-1 {{RISC-V type '__rvv_bfloat16m2_t' requires the 'zvfbfmin' extension}}
  // expected-error@-2 {{builtin requires at least one of the following extensions: zvfbfmin}}
}

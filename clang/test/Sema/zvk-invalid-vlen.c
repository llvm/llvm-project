// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +experimental-zvkned \
// RUN:   -target-feature +experimental-zvksh %s -fsyntax-only -verify

#include <riscv_vector.h>

void test_vaeskf1_tu(vuint32mf2_t vd, vuint32mf2_t vs2, size_t vl) {
  __riscv_vaeskf1_vi_u32mf2_tu(vd, vs2, 0, vl); // expected-error {{RISC-V type 'vuint32mf2_t' (aka '__rvv_uint32mf2_t') requires the 'zvl256b' extension}}
}

void test_vsm3c_tu(vuint32mf2_t vd, vuint32mf2_t vs2, size_t vl) {
  __riscv_vsm3c_vi_u32mf2_tu(vd, vs2, 0, vl); // expected-error {{RISC-V type 'vuint32mf2_t' (aka '__rvv_uint32mf2_t') requires the 'zvl512b' extension}}
}

void test_vaeskf1(vuint32mf2_t vs2, size_t vl) {
  __riscv_vaeskf1_vi_u32mf2(vs2, 0, vl); // expected-error {{RISC-V type 'vuint32mf2_t' (aka '__rvv_uint32mf2_t') requires the 'zvl256b' extension}}
}

void test_vaesdf(vuint32mf2_t vd, vuint32mf2_t vs2, size_t vl) {
  __riscv_vaesdf_vv_u32mf2(vd, vs2, vl); // expected-error {{RISC-V type 'vuint32mf2_t' (aka '__rvv_uint32mf2_t') requires the 'zvl256b' extension}}
}

void test_vaesdf_vs(vuint32m2_t vd, vuint32mf2_t vs2, size_t vl) {
  __riscv_vaesdf_vs_u32mf2_u32m2(vd, vs2, vl); // expected-error {{RISC-V type 'vuint32mf2_t' (aka '__rvv_uint32mf2_t') requires the 'zvl256b' extension}}
}

// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv32 -target-feature +v %s -fsyntax-only -verify

#include <andes_vector.h>

vfloat32mf2_t test_nds_vfwcvt_s_bf16_f32mf2(vbfloat16mf4_t vs2, size_t vl) {
  // expected-error@-1 {{RISC-V type 'vbfloat16mf4_t' (aka '__rvv_bfloat16mf4_t') requires the 'zvfbfmin or xandesvbfhcvt' extension}}
  return __riscv_nds_vfwcvt_s_bf16_f32mf2(vs2, vl); // expected-error {{RISC-V type '__rvv_bfloat16mf4_t' requires the 'zvfbfmin or xandesvbfhcvt' extension}}
  // expected-error@-1 {{RISC-V type 'vbfloat16mf4_t' (aka '__rvv_bfloat16mf4_t') requires the 'zvfbfmin or xandesvbfhcvt' extension}}
  // expected-error@-2 {{builtin requires at least one of the following extensions: xandesvbfhcvt}}
}

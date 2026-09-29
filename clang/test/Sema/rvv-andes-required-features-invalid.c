// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv32 -target-feature +v %s -fsyntax-only -verify

#include <andes_vector.h>

vfloat32mf2_t test_nds_vfwcvt_s_bf16_f32mf2(vbfloat16mf4_t vs2, size_t vl) {
  // expected-error@-1 {{RISC-V type 'vbfloat16mf4_t' (aka '__rvv_bfloat16mf4_t') requires the 'zvfbfmin or xandesvbfhcvt' extension}}
  return __riscv_nds_vfwcvt_s_bf16_f32mf2(vs2, vl); // expected-error {{RISC-V type '__rvv_bfloat16mf4_t' requires the 'zvfbfmin or xandesvbfhcvt' extension}}
  // expected-error@-1 {{RISC-V type 'vbfloat16mf4_t' (aka '__rvv_bfloat16mf4_t') requires the 'zvfbfmin or xandesvbfhcvt' extension}}
  // expected-error@-2 {{builtin requires at least one of the following extensions: xandesvbfhcvt}}
}

vfloat16mf4_t test_nds_vfpmadb_vf_f16mf4(vfloat16mf4_t op1, float op2, size_t vl) {
  // expected-error@-1 {{RISC-V type 'vfloat16mf4_t' (aka '__rvv_float16mf4_t') requires the 'zvfh, zvfhmin or xandesvpackfph' extension}}
  // expected-error@-2 {{RISC-V type 'vfloat16mf4_t' (aka '__rvv_float16mf4_t') requires the 'zvfh, zvfhmin or xandesvpackfph' extension}}
  return __riscv_nds_vfpmadb_vf_f16mf4(op1, op2, vl); // expected-error {{RISC-V type '__rvv_float16mf4_t' requires the 'zvfh, zvfhmin or xandesvpackfph' extension}}
  // expected-error@-1 {{RISC-V type '__rvv_float16mf4_t' requires the 'zvfh, zvfhmin or xandesvpackfph' extension}}
  // expected-error@-2 {{RISC-V type 'vfloat16mf4_t' (aka '__rvv_float16mf4_t') requires the 'zvfh, zvfhmin or xandesvpackfph' extension}}
  // expected-error@-3 {{builtin requires at least one of the following extensions: xandesvpackfph}}
}

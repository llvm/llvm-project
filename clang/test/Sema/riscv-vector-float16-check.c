// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh \
// RUN:   -disable-O0-optnone -o - -fsyntax-only %s -verify 
// REQUIRES: riscv-registered-target
#include <riscv_vector.h>

vfloat16m1_t foo() { /* expected-error {{RISC-V type 'vfloat16m1_t' (aka '__rvv_float16m1_t') requires the 'zvfh' extension}} */
  vfloat16m1_t f16m1; /* expected-error {{RISC-V type 'vfloat16m1_t' (aka '__rvv_float16m1_t') requires the 'zvfh' extension}} */

  (void)f16m1; /* expected-error {{RISC-V type 'vfloat16m1_t' (aka '__rvv_float16m1_t') requires the 'zvfh' extension}} */

  return f16m1; /* expected-error {{RISC-V type 'vfloat16m1_t' (aka '__rvv_float16m1_t') requires the 'zvfh' extension}} */
}

vfloat16m1x2_t bar() { /* expected-error {{RISC-V type 'vfloat16m1x2_t' (aka '__rvv_float16m1x2_t') requires the 'zvfh' extension}} */
  vfloat16m1x2_t f16m1x2; /* expected-error {{RISC-V type 'vfloat16m1x2_t' (aka '__rvv_float16m1x2_t') requires the 'zvfh' extension}} */

  (void)f16m1x2; /* expected-error {{RISC-V type 'vfloat16m1x2_t' (aka '__rvv_float16m1x2_t') requires the 'zvfh' extension}} */

  return f16m1x2; /* expected-error {{RISC-V type 'vfloat16m1x2_t' (aka '__rvv_float16m1x2_t') requires the 'zvfh' extension}} */
}

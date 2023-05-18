// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +zve32x -target-feature +zfh \
// RUN:   -disable-O0-optnone -o - -fsyntax-only %s -verify 
// REQUIRES: riscv-registered-target
#include <riscv_vector.h>

vfloat32m1_t foo() { /* expected-error {{RISC-V type 'vfloat32m1_t' (aka '__rvv_float32m1_t') requires the 'zve32f' extension}} */
  vfloat32m1_t f32m1; /* expected-error {{RISC-V type 'vfloat32m1_t' (aka '__rvv_float32m1_t') requires the 'zve32f' extension}} */

  (void)f32m1; /* expected-error {{RISC-V type 'vfloat32m1_t' (aka '__rvv_float32m1_t') requires the 'zve32f' extension}} */

  return f32m1; /* expected-error {{RISC-V type 'vfloat32m1_t' (aka '__rvv_float32m1_t') requires the 'zve32f' extension}} */
}

vfloat32m1x2_t bar() { /* expected-error {{RISC-V type 'vfloat32m1x2_t' (aka '__rvv_float32m1x2_t') requires the 'zve32f' extension}} */
  vfloat32m1x2_t f32m1x2; /* expected-error {{RISC-V type 'vfloat32m1x2_t' (aka '__rvv_float32m1x2_t') requires the 'zve32f' extension}} */

  (void)f32m1x2; /* expected-error {{RISC-V type 'vfloat32m1x2_t' (aka '__rvv_float32m1x2_t') requires the 'zve32f' extension}} */

  return f32m1x2; /* expected-error {{RISC-V type 'vfloat32m1x2_t' (aka '__rvv_float32m1x2_t') requires the 'zve32f' extension}} */
}

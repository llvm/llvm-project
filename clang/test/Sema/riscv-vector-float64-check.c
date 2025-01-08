// RUN: %clang_cc1 -triple riscv64 -target-feature +zve32x \
// RUN:   -disable-O0-optnone -o - -fsyntax-only %s -verify 
// REQUIRES: riscv-registered-target
#include <riscv_vector.h>

vfloat64m1_t foo() { /* expected-error {{RISC-V type 'vfloat64m1_t' (aka '__rvv_float64m1_t') requires the 'zve64d' extension}} */
  vfloat64m1_t f64m1; /* expected-error {{RISC-V type 'vfloat64m1_t' (aka '__rvv_float64m1_t') requires the 'zve64d' extension}} */

  (void)f64m1; /* expected-error {{RISC-V type 'vfloat64m1_t' (aka '__rvv_float64m1_t') requires the 'zve64d' extension}} */

  return f64m1; /* expected-error {{RISC-V type 'vfloat64m1_t' (aka '__rvv_float64m1_t') requires the 'zve64d' extension}} */
}

vfloat64m1x2_t bar() { /* expected-error {{RISC-V type 'vfloat64m1x2_t' (aka '__rvv_float64m1x2_t') requires the 'zve64d' extension}} */
  vfloat64m1x2_t f64m1x2; /* expected-error {{RISC-V type 'vfloat64m1x2_t' (aka '__rvv_float64m1x2_t') requires the 'zve64d' extension}} */

  (void)f64m1x2; /* expected-error {{RISC-V type 'vfloat64m1x2_t' (aka '__rvv_float64m1x2_t') requires the 'zve64d' extension}} */

  return f64m1x2; /* expected-error {{RISC-V type 'vfloat64m1x2_t' (aka '__rvv_float64m1x2_t') requires the 'zve64d' extension}} */
}

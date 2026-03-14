// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +zve32x -target-feature +zfh \
// RUN:   -disable-O0-optnone -o - -fsyntax-only %s -verify 
// REQUIRES: riscv-registered-target
#include <riscv_vector.h>

vint64m1_t foo() { /* expected-error {{RISC-V type 'vint64m1_t' (aka '__rvv_int64m1_t') requires the 'zve64x' extension}} */
  vint64m1_t i64m1; /* expected-error {{RISC-V type 'vint64m1_t' (aka '__rvv_int64m1_t') requires the 'zve64x' extension}} */

  (void)i64m1; /* expected-error {{RISC-V type 'vint64m1_t' (aka '__rvv_int64m1_t') requires the 'zve64x' extension}} */

  return i64m1; /* expected-error {{RISC-V type 'vint64m1_t' (aka '__rvv_int64m1_t') requires the 'zve64x' extension}} */
}

vint64m1x2_t bar() { /* expected-error {{RISC-V type 'vint64m1x2_t' (aka '__rvv_int64m1x2_t') requires the 'zve64x' extension}} */
  vint64m1x2_t i64m1x2; /* expected-error {{RISC-V type 'vint64m1x2_t' (aka '__rvv_int64m1x2_t') requires the 'zve64x' extension}} */

  (void)i64m1x2; /* expected-error {{RISC-V type 'vint64m1x2_t' (aka '__rvv_int64m1x2_t') requires the 'zve64x' extension}} */

  return i64m1x2; /* expected-error {{RISC-V type 'vint64m1x2_t' (aka '__rvv_int64m1x2_t') requires the 'zve64x' extension}} */
}

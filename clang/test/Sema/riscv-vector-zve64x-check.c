// RUN: %clang_cc1 -triple riscv64 \
// RUN:   -target-feature +zve32x -disable-O0-optnone -o - \
// RUN:   -fsyntax-only %s -verify 
// REQUIRES: riscv-registered-target

  // (ELEN, LMUL) pairs of (8, mf8), (16, mf4), (32, mf2), (64, m1) is not in zve32*
  // available when ELEN is smaller than 64.

__rvv_int8mf8_t foo8() { /* expected-error {{RISC-V type '__rvv_int8mf8_t' requires the 'zve64x' extension}} */
  __rvv_int8mf8_t i8mf8; /* expected-error {{RISC-V type '__rvv_int8mf8_t' requires the 'zve64x' extension}} */

  (void)i8mf8; /* expected-error {{RISC-V type '__rvv_int8mf8_t' requires the 'zve64x' extension}} */

  return i8mf8; /* expected-error {{RISC-V type '__rvv_int8mf8_t' requires the 'zve64x' extension}} */
}

__rvv_int16mf4_t foo16() { /* expected-error {{RISC-V type '__rvv_int16mf4_t' requires the 'zve64x' extension}} */
  __rvv_int16mf4_t i16mf4; /* expected-error {{RISC-V type '__rvv_int16mf4_t' requires the 'zve64x' extension}} */

  (void)i16mf4; /* expected-error {{RISC-V type '__rvv_int16mf4_t' requires the 'zve64x' extension}} */

  return i16mf4; /* expected-error {{RISC-V type '__rvv_int16mf4_t' requires the 'zve64x' extension}} */
}

__rvv_int32mf2_t foo32() { /* expected-error {{RISC-V type '__rvv_int32mf2_t' requires the 'zve64x' extension}} */
  __rvv_int32mf2_t i32mf2; /* expected-error {{RISC-V type '__rvv_int32mf2_t' requires the 'zve64x' extension}} */

  (void)i32mf2; /* expected-error {{RISC-V type '__rvv_int32mf2_t' requires the 'zve64x' extension}} */

  return i32mf2; /* expected-error {{RISC-V type '__rvv_int32mf2_t' requires the 'zve64x' extension}} */
}

__rvv_int64m1_t foo64() { /* expected-error {{RISC-V type '__rvv_int64m1_t' requires the 'zve64x' extension}} */
  __rvv_int64m1_t i64m1; /* expected-error {{RISC-V type '__rvv_int64m1_t' requires the 'zve64x' extension}} */

  (void)i64m1; /* expected-error {{RISC-V type '__rvv_int64m1_t' requires the 'zve64x' extension}} */

  return i64m1; /* expected-error {{RISC-V type '__rvv_int64m1_t' requires the 'zve64x' extension}} */
}

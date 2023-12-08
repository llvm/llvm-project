// RUN: %clang_cc1 -triple riscv64 \
// RUN:   -disable-O0-optnone -o - -fsyntax-only %s -verify 
// REQUIRES: riscv-registered-target

__rvv_int8m1_t foo8() { /* expected-error {{RISC-V type '__rvv_int8m1_t' requires the 'zve32x' extension}} */
  __rvv_int8m1_t i8m1; /* expected-error {{RISC-V type '__rvv_int8m1_t' requires the 'zve32x' extension}} */

  (void)i8m1; /* expected-error {{RISC-V type '__rvv_int8m1_t' requires the 'zve32x' extension}} */

  return i8m1; /* expected-error {{RISC-V type '__rvv_int8m1_t' requires the 'zve32x' extension}} */
}

__rvv_int16m1_t foo16() { /* expected-error {{RISC-V type '__rvv_int16m1_t' requires the 'zve32x' extension}} */
  __rvv_int16m1_t i16m1; /* expected-error {{RISC-V type '__rvv_int16m1_t' requires the 'zve32x' extension}} */

  (void)i16m1; /* expected-error {{RISC-V type '__rvv_int16m1_t' requires the 'zve32x' extension}} */

  return i16m1; /* expected-error {{RISC-V type '__rvv_int16m1_t' requires the 'zve32x' extension}} */
}

__rvv_int32m1_t foo32() { /* expected-error {{RISC-V type '__rvv_int32m1_t' requires the 'zve32x' extension}} */
  __rvv_int32m1_t i32m1; /* expected-error {{RISC-V type '__rvv_int32m1_t' requires the 'zve32x' extension}} */

  (void)i32m1; /* expected-error {{RISC-V type '__rvv_int32m1_t' requires the 'zve32x' extension}} */

  return i32m1; /* expected-error {{RISC-V type '__rvv_int32m1_t' requires the 'zve32x' extension}} */
}

__rvv_int8m1x2_t bar8() { /* expected-error {{RISC-V type '__rvv_int8m1x2_t' requires the 'zve32x' extension}} */
  __rvv_int8m1x2_t i8m1x2; /* expected-error {{RISC-V type '__rvv_int8m1x2_t' requires the 'zve32x' extension}} */

  (void)i8m1x2; /* expected-error {{RISC-V type '__rvv_int8m1x2_t' requires the 'zve32x' extension}} */

  return i8m1x2; /* expected-error {{RISC-V type '__rvv_int8m1x2_t' requires the 'zve32x' extension}} */
}

__rvv_int16m1x2_t bar16() { /* expected-error {{RISC-V type '__rvv_int16m1x2_t' requires the 'zve32x' extension}} */
  __rvv_int16m1x2_t i16m1x2; /* expected-error {{RISC-V type '__rvv_int16m1x2_t' requires the 'zve32x' extension}} */

  (void)i16m1x2; /* expected-error {{RISC-V type '__rvv_int16m1x2_t' requires the 'zve32x' extension}} */

  return i16m1x2; /* expected-error {{RISC-V type '__rvv_int16m1x2_t' requires the 'zve32x' extension}} */
}

__rvv_int32m1x2_t bar32() { /* expected-error {{RISC-V type '__rvv_int32m1x2_t' requires the 'zve32x' extension}} */
  __rvv_int32m1x2_t i32m1x2; /* expected-error {{RISC-V type '__rvv_int32m1x2_t' requires the 'zve32x' extension}} */

  (void)i32m1x2; /* expected-error {{RISC-V type '__rvv_int32m1x2_t' requires the 'zve32x' extension}} */

  return i32m1x2; /* expected-error {{RISC-V type '__rvv_int32m1x2_t' requires the 'zve32x' extension}} */
}

__rvv_bool1_t vbool1 () { /* expected-error {{RISC-V type '__rvv_bool1_t' requires the 'zve32x' extension}} */
  __rvv_bool1_t b1; /* expected-error {{RISC-V type '__rvv_bool1_t' requires the 'zve32x' extension}} */

  (void)b1; /* expected-error {{RISC-V type '__rvv_bool1_t' requires the 'zve32x' extension}} */

  return b1; /* expected-error {{RISC-V type '__rvv_bool1_t' requires the 'zve32x' extension}} */
}

__rvv_bool2_t vbool2 () { /* expected-error {{RISC-V type '__rvv_bool2_t' requires the 'zve32x' extension}} */
  __rvv_bool2_t b2; /* expected-error {{RISC-V type '__rvv_bool2_t' requires the 'zve32x' extension}} */

  (void)b2; /* expected-error {{RISC-V type '__rvv_bool2_t' requires the 'zve32x' extension}} */

  return b2; /* expected-error {{RISC-V type '__rvv_bool2_t' requires the 'zve32x' extension}} */
}

__rvv_bool4_t vbool4 () { /* expected-error {{RISC-V type '__rvv_bool4_t' requires the 'zve32x' extension}} */
  __rvv_bool4_t b4; /* expected-error {{RISC-V type '__rvv_bool4_t' requires the 'zve32x' extension}} */

  (void)b4; /* expected-error {{RISC-V type '__rvv_bool4_t' requires the 'zve32x' extension}} */

  return b4; /* expected-error {{RISC-V type '__rvv_bool4_t' requires the 'zve32x' extension}} */
}

__rvv_bool8_t vbool8 () { /* expected-error {{RISC-V type '__rvv_bool8_t' requires the 'zve32x' extension}} */
  __rvv_bool8_t b8; /* expected-error {{RISC-V type '__rvv_bool8_t' requires the 'zve32x' extension}} */

  (void)b8; /* expected-error {{RISC-V type '__rvv_bool8_t' requires the 'zve32x' extension}} */

  return b8; /* expected-error {{RISC-V type '__rvv_bool8_t' requires the 'zve32x' extension}} */
}

__rvv_bool16_t vbool16 () { /* expected-error {{RISC-V type '__rvv_bool16_t' requires the 'zve32x' extension}} */
  __rvv_bool16_t b16; /* expected-error {{RISC-V type '__rvv_bool16_t' requires the 'zve32x' extension}} */

  (void)b16; /* expected-error {{RISC-V type '__rvv_bool16_t' requires the 'zve32x' extension}} */

  return b16; /* expected-error {{RISC-V type '__rvv_bool16_t' requires the 'zve32x' extension}} */
}

__rvv_bool32_t vbool32 () { /* expected-error {{RISC-V type '__rvv_bool32_t' requires the 'zve32x' extension}} */
  __rvv_bool32_t b32; /* expected-error {{RISC-V type '__rvv_bool32_t' requires the 'zve32x' extension}} */

  (void)b32; /* expected-error {{RISC-V type '__rvv_bool32_t' requires the 'zve32x' extension}} */

  return b32; /* expected-error {{RISC-V type '__rvv_bool32_t' requires the 'zve32x' extension}} */
}

__rvv_bool64_t vbool64 () { /* expected-error {{RISC-V type '__rvv_bool64_t' requires the 'zve32x' extension}} */
  __rvv_bool64_t b64; /* expected-error {{RISC-V type '__rvv_bool64_t' requires the 'zve32x' extension}} */

  (void)b64; /* expected-error {{RISC-V type '__rvv_bool64_t' requires the 'zve32x' extension}} */

  return b64; /* expected-error {{RISC-V type '__rvv_bool64_t' requires the 'zve32x' extension}} */
}

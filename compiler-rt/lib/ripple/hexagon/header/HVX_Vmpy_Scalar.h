//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//
// Public API declarations for the HVX scalar-broadcast multiply runtime
// library.
//
// The Ripple compiler pass replaces calls to hvx_*_vmpy_* / hvx_*_vmpyacc_*
// with the corresponding ripple_ew_pure_hvx_* implementations from
// HVX_Vmpy_Scalar.cc.
//
// The polymorphic hvx_vmpy* / hvx_vmpyacc* entry points follow the same
// pattern as hvx_vdeal in HVX_Vdeal.h: typed extern "C" declarations via
// _decl_* macros, then C _Generic dispatch and C++ overloaded static inlines.
//
// Note on C _Generic dispatch:
//   For hvx_vmpy, hvx_vmpy_noshuff, hvx_vmpyacc, and hvx_vmpyacc_noshuff,
//   the scalar (last) argument 'b' uniquely identifies the intrinsic:
//     signed char    -> i16 output (u8i8 variant)
//     unsigned char  -> u16 output (u8u8 variant)
//     unsigned short -> u32 output (u16u16 variant)
//     short          -> i32 output (i16i16 variant)
//   Plain 'char' dispatches to the unsigned char / u8u8 variant because
//   char is always unsigned on Hexagon.

#pragma once
#include "__ripple_vec.h"
#include <stddef.h>

// Use basic C types instead of stdint types: on Hexagon, stdint does not
// define a type for plain 'int' (4 bytes), so stdint types cannot cover all
// standard C types.  Each declaration below carries a static_assert that
// verifies the assumed size on the target.

#ifdef __cplusplus
extern "C" {
#endif

static_assert(sizeof(signed char) == 1 && sizeof(short) == 2,
              "HVX_Vmpy_Scalar: unexpected char/short size");
static_assert(sizeof(int) == 4, "HVX_Vmpy_Scalar: unexpected int size");
static_assert(sizeof(unsigned int) == 4,
              "HVX_Vmpy_Scalar: unexpected unsigned int size");

// ---------------------------------------------------------------------------
// vmpy: vector * scalar (widening, odd/even interleaved output)
// ---------------------------------------------------------------------------
#define _decl_hvx_vmpy(cOutType, outType, cVecType, vecType, cScType, scType)  \
  __attribute__((always_inline)) extern cOutType                               \
      hvx_##outType##_vmpy_##vecType##scType(cVecType a, cScType b)

_decl_hvx_vmpy(short, i16, unsigned char, u8, signed char,
               i8); // Q6_Wh_vmpy_VubRb
_decl_hvx_vmpy(unsigned short, u16, unsigned char, u8, unsigned char,
               u8);                               // Q6_Wuh_vmpy_VubRub
_decl_hvx_vmpy(int, i32, short, i16, short, i16); // Q6_Ww_vmpy_VhRh

// ---------------------------------------------------------------------------
// vmpy_s1_sat / vmpy_s1_rnd_sat: non-widening, sequential output, <<1 + sat
// ---------------------------------------------------------------------------
#define _decl_hvx_vmpy_s1_sat(cType, type)                                     \
  __attribute__((always_inline)) extern cType                                  \
      hvx_##type##_vmpy_##type##type##_s1_sat(cType v, cType r)

#define _decl_hvx_vmpy_s1_rnd_sat(cType, type)                                 \
  __attribute__((always_inline)) extern cType                                  \
      hvx_##type##_vmpy_##type##type##_s1_rnd_sat(cType v, cType r)

_decl_hvx_vmpy_s1_sat(short, i16);     // Q6_Vh_vmpy_VhRh_s1_sat
_decl_hvx_vmpy_s1_rnd_sat(short, i16); // Q6_Vh_vmpy_VhRh_s1_rnd_sat

// ---------------------------------------------------------------------------
// vmpy_noshuff: widening, deinterleaved to sequential output order
// ---------------------------------------------------------------------------
#define _decl_hvx_vmpy_noshuff(cOutType, outType, cVecType, vecType, cScType,  \
                               scType)                                         \
  __attribute__((always_inline)) extern cOutType                               \
      hvx_##outType##_vmpy_##vecType##scType##_noshuff(cVecType a, cScType b)

_decl_hvx_vmpy_noshuff(short, i16, unsigned char, u8, signed char,
                       i8); // Q6_Wh_vmpy_VubRb   + vdeal
_decl_hvx_vmpy_noshuff(unsigned short, u16, unsigned char, u8, unsigned char,
                       u8); // Q6_Wuh_vmpy_VubRub + vdeal
_decl_hvx_vmpy_noshuff(int, i32, short, i16, short,
                       i16); // Q6_Ww_vmpy_VhRh    + vdeal

// ---------------------------------------------------------------------------
// vmpyacc: accumulate + vector * scalar (widening, odd/even interleaved output)
// NOTE: accumulator must be in odd/even interleaved order.
// ---------------------------------------------------------------------------
#define _decl_hvx_vmpyacc(cOutType, outType, cAccType, accType, cVecType,      \
                          vecType, cScType, scType)                            \
  __attribute__((always_inline)) extern cOutType                               \
      hvx_##outType##_vmpyacc_##accType##vecType##scType(                      \
          cAccType acc, cVecType a, cScType b)

_decl_hvx_vmpyacc(short, i16, short, i16, unsigned char, u8, signed char,
                  i8); // Q6_Wh_vmpyacc_WhVubRb
_decl_hvx_vmpyacc(unsigned short, u16, unsigned short, u16, unsigned char, u8,
                  unsigned char, u8); // Q6_Wuh_vmpyacc_WuhVubRub
_decl_hvx_vmpyacc(unsigned int, u32, unsigned int, u32, unsigned short, u16,
                  unsigned short, u16); // Q6_Wuw_vmpyacc_WuwVuhRuh
_decl_hvx_vmpyacc(int, i32, int, i32, short, i16, short,
                  i16); // Q6_Ww_vmpyacc_WwVhRh

// ---------------------------------------------------------------------------
// vmpyacc_sat: accumulate + vector * scalar, saturating (odd/even interleaved)
// NOTE: accumulator must be in odd/even interleaved order.
// ---------------------------------------------------------------------------
#define _decl_hvx_vmpyacc_sat(cType, type, cVecType, vecType, cScType, scType) \
  __attribute__((always_inline)) extern cType                                  \
      hvx_##type##_vmpyacc_##type##vecType##scType##_sat(                      \
          cType acc, cVecType a, cScType b)

_decl_hvx_vmpyacc_sat(int, i32, short, i16, short,
                      i16); // Q6_Ww_vmpyacc_WwVhRh_sat

// ---------------------------------------------------------------------------
// vmpyacc_noshuff: widening, deinterleaved to sequential output order
// NOTE: accumulator must be in sequential order.
// ---------------------------------------------------------------------------
#define _decl_hvx_vmpyacc_noshuff(cOutType, outType, cAccType, accType,        \
                                  cVecType, vecType, cScType, scType)          \
  __attribute__((always_inline)) extern cOutType                               \
      hvx_##outType##_vmpyacc_##accType##vecType##scType##_noshuff(            \
          cAccType acc, cVecType a, cScType b)

_decl_hvx_vmpyacc_noshuff(short, i16, short, i16, unsigned char, u8,
                          signed char, i8); // Q6_Wh_vmpyacc_WhVubRb   + vdeal
_decl_hvx_vmpyacc_noshuff(unsigned short, u16, unsigned short, u16,
                          unsigned char, u8, unsigned char,
                          u8); // Q6_Wuh_vmpyacc_WuhVubRub + vdeal
_decl_hvx_vmpyacc_noshuff(unsigned int, u32, unsigned int, u32, unsigned short,
                          u16, unsigned short,
                          u16); // Q6_Wuw_vmpyacc_WuwVuhRuh + vdeal
_decl_hvx_vmpyacc_noshuff(int, i32, int, i32, short, i16, short,
                          i16); // Q6_Ww_vmpyacc_WwVhRh     + vdeal

// ---------------------------------------------------------------------------
// vmpyacc_sat_noshuff: saturating, deinterleaved to sequential output order
// NOTE: accumulator must be in sequential order.
// ---------------------------------------------------------------------------
#define _decl_hvx_vmpyacc_sat_noshuff(cType, type, cVecType, vecType, cScType, \
                                      scType)                                  \
  __attribute__((always_inline)) extern cType                                  \
      hvx_##type##_vmpyacc_##type##vecType##scType##_sat_noshuff(              \
          cType acc, cVecType a, cScType b)

_decl_hvx_vmpyacc_sat_noshuff(int, i32, short, i16, short,
                              i16); // Q6_Ww_vmpyacc_WwVhRh_sat + vdeal

// ---------------------------------------------------------------------------
// Special case: u16 vector x two packed u16 scalars -> u32
// r encodes two u16 values packed into a u32 (used by Q6_Wuw_vmpy_VuhRuh).
// ---------------------------------------------------------------------------
// OUTPUT: odd/even interleaved.
__attribute__((always_inline)) extern unsigned int
hvx_u32_vmpy_u16_2u16(unsigned short in, unsigned int r);
// OUTPUT: sequential (deinterleaved via vdeal).
__attribute__((always_inline)) extern unsigned int
hvx_u32_vmpy_u16_2u16_noshuff(unsigned short in, unsigned int r);

#undef _decl_hvx_vmpy
#undef _decl_hvx_vmpy_s1_sat
#undef _decl_hvx_vmpy_s1_rnd_sat
#undef _decl_hvx_vmpy_noshuff
#undef _decl_hvx_vmpyacc
#undef _decl_hvx_vmpyacc_sat
#undef _decl_hvx_vmpyacc_noshuff
#undef _decl_hvx_vmpyacc_sat_noshuff

#ifdef __cplusplus
} // extern "C"
#endif

// =============================================================================
// Polymorphic wrappers
// =============================================================================

#ifndef __cplusplus

// --- C: _Generic dispatch ---

// hvx_vmpy(a, b) — widening multiply, odd/even interleaved output.
// Dispatch is on the scalar (second) argument 'b'.
// Plain 'char' dispatches to the u8u8 variant (char is always unsigned on
// Hexagon).
#define hvx_vmpy(a, b)                                                         \
  _Generic((b),                                                                \
      char: hvx_u16_vmpy_u8u8((unsigned char)(a), (unsigned char)(b)),         \
      signed char: hvx_i16_vmpy_u8i8((unsigned char)(a), (b)),                 \
      unsigned char: hvx_u16_vmpy_u8u8((unsigned char)(a), (b)),               \
      short: hvx_i32_vmpy_i16i16((short)(a), (b)))

// hvx_vmpy_noshuff(a, b) — widening multiply, sequential (deinterleaved)
// output. Dispatch is on the scalar (second) argument 'b'.
#define hvx_vmpy_noshuff(a, b)                                                 \
  _Generic((b),                                                                \
      char: hvx_u16_vmpy_u8u8_noshuff((unsigned char)(a), (unsigned char)(b)), \
      signed char: hvx_i16_vmpy_u8i8_noshuff((unsigned char)(a), (b)),         \
      unsigned char: hvx_u16_vmpy_u8u8_noshuff((unsigned char)(a), (b)),       \
      short: hvx_i32_vmpy_i16i16_noshuff((short)(a), (b)))

// hvx_vmpy_s1_sat(v, r) — non-widening <<1 saturating multiply.
#define hvx_vmpy_s1_sat(v, r)                                                  \
  _Generic((v), short: hvx_i16_vmpy_i16i16_s1_sat((v), (r)))

// hvx_vmpy_s1_rnd_sat(v, r) — non-widening <<1 rounding saturating multiply.
#define hvx_vmpy_s1_rnd_sat(v, r)                                              \
  _Generic((v), short: hvx_i16_vmpy_i16i16_s1_rnd_sat((v), (r)))

// hvx_vmpyacc(acc, a, b) — accumulate + widening multiply, interleaved output.
// Dispatch is on the scalar (third) argument 'b'.
// Plain 'char' dispatches to the u8u8 variant (char is always unsigned on
// Hexagon).
#define hvx_vmpyacc(acc, a, b)                                                 \
  _Generic((b),                                                                \
      char: hvx_u16_vmpyacc_u16u8u8((unsigned short)(acc), (unsigned char)(a), \
                                    (unsigned char)(b)),                       \
      signed char: hvx_i16_vmpyacc_i16u8i8((short)(acc), (unsigned char)(a),   \
                                           (b)),                               \
      unsigned char: hvx_u16_vmpyacc_u16u8u8((unsigned short)(acc),            \
                                             (unsigned char)(a), (b)),         \
      unsigned short: hvx_u32_vmpyacc_u32u16u16((unsigned int)(acc),           \
                                                (unsigned short)(a), (b)),     \
      short: hvx_i32_vmpyacc_i32i16i16((int)(acc), (short)(a), (b)))

// hvx_vmpyacc_noshuff(acc, a, b) — accumulate + widening multiply, sequential.
// Dispatch is on the scalar (third) argument 'b'.
#define hvx_vmpyacc_noshuff(acc, a, b)                                         \
  _Generic((b),                                                                \
      char: hvx_u16_vmpyacc_u16u8u8_noshuff(                                   \
               (unsigned short)(acc), (unsigned char)(a), (unsigned char)(b)), \
      signed char: hvx_i16_vmpyacc_i16u8i8_noshuff((short)(acc),               \
                                                   (unsigned char)(a), (b)),   \
      unsigned char: hvx_u16_vmpyacc_u16u8u8_noshuff((unsigned short)(acc),    \
                                                     (unsigned char)(a), (b)), \
      unsigned short: hvx_u32_vmpyacc_u32u16u16_noshuff(                       \
               (unsigned int)(acc), (unsigned short)(a), (b)),                 \
      short: hvx_i32_vmpyacc_i32i16i16_noshuff((int)(acc), (short)(a), (b)))

// hvx_vmpyacc_sat(acc, a, b) — saturating accumulate + widening multiply.
#define hvx_vmpyacc_sat(acc, a, b)                                             \
  _Generic((b),                                                                \
      short: hvx_i32_vmpyacc_i32i16i16_sat((int)(acc), (short)(a), (b)))

// hvx_vmpyacc_sat_noshuff(acc, a, b) — saturating, sequential output.
#define hvx_vmpyacc_sat_noshuff(acc, a, b)                                     \
  _Generic((b),                                                                \
      short: hvx_i32_vmpyacc_i32i16i16_sat_noshuff((int)(acc), (short)(a),     \
                                                   (b)))

#else // __cplusplus

// --- C++: overloaded static inline functions ---

// hvx_vmpy — widening multiply, odd/even interleaved output.
// char is always unsigned on Hexagon — dispatches to the u8u8 variant.
// Both signed-char and unsigned-char overloads are provided; the second
// argument type resolves the ambiguity for explicit signed/unsigned char.
[[gnu::always_inline, gnu::unused]] static unsigned short hvx_vmpy(char a,
                                                                   char b) {
  return hvx_u16_vmpy_u8u8((unsigned char)a, (unsigned char)b);
}
[[gnu::always_inline, gnu::unused]] static unsigned short
hvx_vmpy(unsigned char a, char b) {
  return hvx_u16_vmpy_u8u8(a, (unsigned char)b);
}
[[gnu::always_inline, gnu::unused]] static short hvx_vmpy(unsigned char a,
                                                          signed char b) {
  return hvx_i16_vmpy_u8i8(a, b);
}
[[gnu::always_inline, gnu::unused]] static unsigned short
hvx_vmpy(unsigned char a, unsigned char b) {
  return hvx_u16_vmpy_u8u8(a, b);
}
[[gnu::always_inline, gnu::unused]] static int hvx_vmpy(short a, short b) {
  return hvx_i32_vmpy_i16i16(a, b);
}

// hvx_vmpy_noshuff — widening multiply, sequential (deinterleaved) output.
[[gnu::always_inline, gnu::unused]] static unsigned short
hvx_vmpy_noshuff(char a, char b) {
  return hvx_u16_vmpy_u8u8_noshuff((unsigned char)a, (unsigned char)b);
}
[[gnu::always_inline, gnu::unused]] static unsigned short
hvx_vmpy_noshuff(unsigned char a, char b) {
  return hvx_u16_vmpy_u8u8_noshuff(a, (unsigned char)b);
}
[[gnu::always_inline, gnu::unused]] static short
hvx_vmpy_noshuff(unsigned char a, signed char b) {
  return hvx_i16_vmpy_u8i8_noshuff(a, b);
}
[[gnu::always_inline, gnu::unused]] static unsigned short
hvx_vmpy_noshuff(unsigned char a, unsigned char b) {
  return hvx_u16_vmpy_u8u8_noshuff(a, b);
}
[[gnu::always_inline, gnu::unused]] static int hvx_vmpy_noshuff(short a,
                                                                short b) {
  return hvx_i32_vmpy_i16i16_noshuff(a, b);
}

// hvx_vmpy_s1_sat — non-widening <<1 saturating multiply.
[[gnu::always_inline, gnu::unused]] static short hvx_vmpy_s1_sat(short v,
                                                                 short r) {
  return hvx_i16_vmpy_i16i16_s1_sat(v, r);
}

// hvx_vmpy_s1_rnd_sat — non-widening <<1 rounding saturating multiply.
[[gnu::always_inline, gnu::unused]] static short hvx_vmpy_s1_rnd_sat(short v,
                                                                     short r) {
  return hvx_i16_vmpy_i16i16_s1_rnd_sat(v, r);
}

// hvx_vmpyacc — accumulate + widening multiply, odd/even interleaved output.
// char is always unsigned on Hexagon — dispatches to the u8u8 variant.
[[gnu::always_inline, gnu::unused]] static unsigned short
hvx_vmpyacc(char acc, char a, char b) {
  return hvx_u16_vmpyacc_u16u8u8((unsigned short)acc, (unsigned char)a,
                                 (unsigned char)b);
}
[[gnu::always_inline, gnu::unused]] static short
hvx_vmpyacc(short acc, unsigned char a, signed char b) {
  return hvx_i16_vmpyacc_i16u8i8(acc, a, b);
}
[[gnu::always_inline, gnu::unused]] static unsigned short
hvx_vmpyacc(short acc, unsigned char a, char b) {
  return hvx_u16_vmpyacc_u16u8u8((unsigned short)acc, a, (unsigned char)b);
}
[[gnu::always_inline, gnu::unused]] static unsigned short
hvx_vmpyacc(unsigned short acc, unsigned char a, unsigned char b) {
  return hvx_u16_vmpyacc_u16u8u8(acc, a, b);
}
[[gnu::always_inline, gnu::unused]] static unsigned int
hvx_vmpyacc(unsigned int acc, unsigned short a, unsigned short b) {
  return hvx_u32_vmpyacc_u32u16u16(acc, a, b);
}
[[gnu::always_inline, gnu::unused]] static int hvx_vmpyacc(int acc, short a,
                                                           short b) {
  return hvx_i32_vmpyacc_i32i16i16(acc, a, b);
}

// hvx_vmpyacc_noshuff — accumulate + widening multiply, sequential output.
[[gnu::always_inline, gnu::unused]] static unsigned short
hvx_vmpyacc_noshuff(char acc, char a, char b) {
  return hvx_u16_vmpyacc_u16u8u8_noshuff((unsigned short)acc, (unsigned char)a,
                                         (unsigned char)b);
}
[[gnu::always_inline, gnu::unused]] static short
hvx_vmpyacc_noshuff(short acc, unsigned char a, signed char b) {
  return hvx_i16_vmpyacc_i16u8i8_noshuff(acc, a, b);
}
[[gnu::always_inline, gnu::unused]] static unsigned short
hvx_vmpyacc_noshuff(short acc, unsigned char a, char b) {
  return hvx_u16_vmpyacc_u16u8u8_noshuff((unsigned short)acc, a,
                                         (unsigned char)b);
}
[[gnu::always_inline, gnu::unused]] static unsigned short
hvx_vmpyacc_noshuff(unsigned short acc, unsigned char a, unsigned char b) {
  return hvx_u16_vmpyacc_u16u8u8_noshuff(acc, a, b);
}
[[gnu::always_inline, gnu::unused]] static unsigned int
hvx_vmpyacc_noshuff(unsigned int acc, unsigned short a, unsigned short b) {
  return hvx_u32_vmpyacc_u32u16u16_noshuff(acc, a, b);
}
[[gnu::always_inline, gnu::unused]] static int
hvx_vmpyacc_noshuff(int acc, short a, short b) {
  return hvx_i32_vmpyacc_i32i16i16_noshuff(acc, a, b);
}

// hvx_vmpyacc_sat — saturating accumulate + widening multiply, interleaved.
[[gnu::always_inline, gnu::unused]] static int hvx_vmpyacc_sat(int acc, short a,
                                                               short b) {
  return hvx_i32_vmpyacc_i32i16i16_sat(acc, a, b);
}

// hvx_vmpyacc_sat_noshuff — saturating accumulate + widening multiply,
// sequential.
[[gnu::always_inline, gnu::unused]] static int
hvx_vmpyacc_sat_noshuff(int acc, short a, short b) {
  return hvx_i32_vmpyacc_i32i16i16_sat_noshuff(acc, a, b);
}

#endif // __cplusplus

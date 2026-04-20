//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//
// Public API declarations for the HVX vector average runtime library.
//
// These declarations expose the Ripple HVX vavg/vnavg functions to user code.
// The Ripple compiler pass replaces calls to hvx_*_vavg_* / hvx_*_vnavg_*
// with the corresponding ripple_ew_pure_hvx_* implementations from HVX_Vavg.cc.
//
// The polymorphic hvx_vavg / hvx_vavg_rnd / hvx_vnavg entry points follow the
// same pattern as hvx_vdeal in HVX_Vdeal.h: typed extern "C" declarations via
// _decl_* macros, then C _Generic dispatch and C++ overloaded static inlines.

#pragma once
#include "__ripple_vec.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Declares: hvx_<outType>_vavg_<inType><inType>
// Truncating average: out[i] = (a[i] + b[i]) >> 1
#define _decl_hvx_vavg(cOutType, outType, cInType, inType)                     \
  __attribute__((always_inline)) extern cOutType                               \
      hvx_##outType##_vavg_##inType##inType(cInType a, cInType b)

// Declares: hvx_<outType>_vavg_<inType><inType>_rnd
// Rounding average: out[i] = (a[i] + b[i] + 1) >> 1
#define _decl_hvx_vavg_rnd(cOutType, outType, cInType, inType)                 \
  __attribute__((always_inline)) extern cOutType                               \
      hvx_##outType##_vavg_##inType##inType##_rnd(cInType a, cInType b)

// Declares: hvx_<outType>_vnavg_<inType><inType>
// Negated average: out[i] = (a[i] - b[i]) >> 1
#define _decl_hvx_vnavg(cOutType, outType, cInType, inType)                    \
  __attribute__((always_inline)) extern cOutType                               \
      hvx_##outType##_vnavg_##inType##inType(cInType a, cInType b)

// --- vavg: truncating average ---
_decl_hvx_vavg(unsigned char, u8, unsigned char, u8);
_decl_hvx_vavg(signed char, i8, signed char, i8);
_decl_hvx_vavg(unsigned short, u16, unsigned short, u16);
_decl_hvx_vavg(short, i16, short, i16);
_decl_hvx_vavg(unsigned int, u32, unsigned int, u32);
_decl_hvx_vavg(int, i32, int, i32);

// --- vavg_rnd: rounding average ---
_decl_hvx_vavg_rnd(unsigned char, u8, unsigned char, u8);
_decl_hvx_vavg_rnd(signed char, i8, signed char, i8);
_decl_hvx_vavg_rnd(unsigned short, u16, unsigned short, u16);
_decl_hvx_vavg_rnd(short, i16, short, i16);
_decl_hvx_vavg_rnd(unsigned int, u32, unsigned int, u32);
_decl_hvx_vavg_rnd(int, i32, int, i32);

// --- vnavg: negated average ---
// Note: the u8 input variant produces a signed i8 output.
_decl_hvx_vnavg(signed char, i8, signed char, i8);
_decl_hvx_vnavg(signed char, i8, unsigned char, u8);
_decl_hvx_vnavg(short, i16, short, i16);
_decl_hvx_vnavg(int, i32, int, i32);

#undef _decl_hvx_vavg
#undef _decl_hvx_vavg_rnd
#undef _decl_hvx_vnavg

#ifdef __cplusplus
} // extern "C"
#endif

// =============================================================================
// Polymorphic wrappers
// =============================================================================

#ifndef __cplusplus

// --- C: _Generic dispatch ---

// hvx_vavg(a, b) — truncating average, dispatches on type of a
#define hvx_vavg(a, b)                                                         \
  _Generic((a),                                                                \
      char: hvx_u8_vavg_u8u8((unsigned char)(a), (unsigned char)(b)),          \
      signed char: hvx_i8_vavg_i8i8((a), (b)),                                 \
      unsigned char: hvx_u8_vavg_u8u8((a), (b)),                               \
      signed short: hvx_i16_vavg_i16i16((a), (b)),                             \
      unsigned short: hvx_u16_vavg_u16u16((a), (b)),                           \
      signed int: hvx_i32_vavg_i32i32((a), (b)),                               \
      unsigned int: hvx_u32_vavg_u32u32((a), (b)),                             \
      signed long: hvx_i32_vavg_i32i32((int)(a), (int)(b)),                    \
      unsigned long: hvx_u32_vavg_u32u32((unsigned int)(a),                    \
                                         (unsigned int)(b)))

// hvx_vavg_rnd(a, b) — rounding average, dispatches on type of a
#define hvx_vavg_rnd(a, b)                                                     \
  _Generic((a),                                                                \
      char: hvx_u8_vavg_u8u8_rnd((unsigned char)(a), (unsigned char)(b)),      \
      signed char: hvx_i8_vavg_i8i8_rnd((a), (b)),                             \
      unsigned char: hvx_u8_vavg_u8u8_rnd((a), (b)),                           \
      signed short: hvx_i16_vavg_i16i16_rnd((a), (b)),                         \
      unsigned short: hvx_u16_vavg_u16u16_rnd((a), (b)),                       \
      signed int: hvx_i32_vavg_i32i32_rnd((a), (b)),                           \
      unsigned int: hvx_u32_vavg_u32u32_rnd((a), (b)),                         \
      signed long: hvx_i32_vavg_i32i32_rnd((int)(a), (int)(b)),                \
      unsigned long: hvx_u32_vavg_u32u32_rnd((unsigned int)(a),                \
                                             (unsigned int)(b)))

// hvx_vnavg(a, b) — negated average, dispatches on type of a.
// For unsigned char inputs the result is signed char (i8).
#define hvx_vnavg(a, b)                                                        \
  _Generic((a),                                                                \
      char: hvx_i8_vnavg_u8u8((unsigned char)(a), (unsigned char)(b)),         \
      signed char: hvx_i8_vnavg_i8i8((a), (b)),                                \
      unsigned char: hvx_i8_vnavg_u8u8((a), (b)),                              \
      signed short: hvx_i16_vnavg_i16i16((a), (b)),                            \
      signed int: hvx_i32_vnavg_i32i32((a), (b)),                              \
      signed long: hvx_i32_vnavg_i32i32((int)(a), (int)(b)))

#else // __cplusplus

// --- C++: overloaded static inline functions ---

#define _decl_hvx_vavg_spec(CT, UI, SZ)                                        \
  [[gnu::always_inline, gnu::unused]] static CT hvx_vavg(CT a, CT b) {         \
    return hvx_##UI##SZ##_vavg_##UI##SZ##UI##SZ(a, b);                         \
  }

#define _decl_hvx_vavg_rnd_spec(CT, UI, SZ)                                    \
  [[gnu::always_inline, gnu::unused]] static CT hvx_vavg_rnd(CT a, CT b) {     \
    return hvx_##UI##SZ##_vavg_##UI##SZ##UI##SZ##_rnd(a, b);                   \
  }

[[gnu::always_inline, gnu::unused]] static char hvx_vavg(char a, char b) {
  return hvx_u8_vavg_u8u8(a, b);
}
_decl_hvx_vavg_spec(signed char, i, 8);
_decl_hvx_vavg_spec(unsigned char, u, 8);
_decl_hvx_vavg_spec(signed short, i, 16);
_decl_hvx_vavg_spec(unsigned short, u, 16);
_decl_hvx_vavg_spec(signed int, i, 32);
_decl_hvx_vavg_spec(unsigned int, u, 32);
// signed/unsigned long are 32-bit on Hexagon — reuse i32/u32 implementations.
[[gnu::always_inline, gnu::unused]] static signed long hvx_vavg(signed long a,
                                                                signed long b) {
  return hvx_i32_vavg_i32i32((int)a, (int)b);
}
[[gnu::always_inline, gnu::unused]] static unsigned long
hvx_vavg(unsigned long a, unsigned long b) {
  return hvx_u32_vavg_u32u32((unsigned int)a, (unsigned int)b);
}

[[gnu::always_inline, gnu::unused]] static char hvx_vavg_rnd(char a, char b) {
  return hvx_u8_vavg_u8u8_rnd(a, b);
}
_decl_hvx_vavg_rnd_spec(signed char, i, 8);
_decl_hvx_vavg_rnd_spec(unsigned char, u, 8);
_decl_hvx_vavg_rnd_spec(signed short, i, 16);
_decl_hvx_vavg_rnd_spec(unsigned short, u, 16);
_decl_hvx_vavg_rnd_spec(signed int, i, 32);
_decl_hvx_vavg_rnd_spec(unsigned int, u, 32);
// signed/unsigned long are 32-bit on Hexagon — reuse i32/u32 implementations.
[[gnu::always_inline, gnu::unused]] static signed long
hvx_vavg_rnd(signed long a, signed long b) {
  return hvx_i32_vavg_i32i32_rnd((int)a, (int)b);
}
[[gnu::always_inline, gnu::unused]] static unsigned long
hvx_vavg_rnd(unsigned long a, unsigned long b) {
  return hvx_u32_vavg_u32u32_rnd((unsigned int)a, (unsigned int)b);
}

#undef _decl_hvx_vavg_spec
#undef _decl_hvx_vavg_rnd_spec

// vnavg: output is always signed; unsigned char input is a distinct overload.
[[gnu::always_inline, gnu::unused]] static signed char hvx_vnavg(char a,
                                                                 char b) {
  return hvx_i8_vnavg_u8u8(a, b);
}
[[gnu::always_inline, gnu::unused]] static signed char
hvx_vnavg(signed char a, signed char b) {
  return hvx_i8_vnavg_i8i8(a, b);
}
[[gnu::always_inline, gnu::unused]] static signed char
hvx_vnavg(unsigned char a, unsigned char b) {
  return hvx_i8_vnavg_u8u8(a, b);
}
[[gnu::always_inline, gnu::unused]] static short hvx_vnavg(short a, short b) {
  return hvx_i16_vnavg_i16i16(a, b);
}
[[gnu::always_inline, gnu::unused]] static int hvx_vnavg(int a, int b) {
  return hvx_i32_vnavg_i32i32(a, b);
}
// signed long is 32-bit on Hexagon — reuse i32 implementation.
[[gnu::always_inline, gnu::unused]] static signed long
hvx_vnavg(signed long a, signed long b) {
  return hvx_i32_vnavg_i32i32((int)a, (int)b);
}

#endif // __cplusplus

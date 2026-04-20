//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//
// Public API declarations for the HVX 32x16 odd-element multiply (vmpyo)
// runtime library.
//
// vmpyo multiplies the odd (upper) 16-bit half of each 32-bit lane of 'm' by
// the corresponding 16-bit lane of 'mc', with a left-shift by 1 and optional
// rounding/saturation.
//
// The Ripple compiler pass replaces calls to hvx_i32_vmpyo_* with the
// corresponding ripple_ew_pure_hvx_* implementations from HVX_Vmpy_x32x16.cc.
//
// The polymorphic hvx_vmpyo_s1_rnd_sat / hvx_vmpyo_s1_sat entry points follow
// the same pattern as hvx_vdeal in HVX_Vdeal.h: typed extern "C" declarations
// via _decl_* macros, then C _Generic dispatch and C++ overloaded static
// inlines.

#pragma once
#include <stddef.h>

// Use basic C types instead of stdint types: on Hexagon, stdint does not
// define a type for plain 'int' (4 bytes), so stdint types cannot cover all
// standard C types.
static_assert(sizeof(int) == 4, "HVX_Vmpy_x32x16: unexpected int size");

#ifdef __cplusplus
extern "C" {
#endif

// out[i] = sat((m[i].hi16 * mc[i] << 1) + rounding) >> 16
#define _decl_hvx_vmpyo_s1_rnd_sat(cType, type, cScType, scType)               \
  __attribute__((always_inline)) extern cType                                  \
      hvx_##type##_vmpyo_##type##scType##_s1_rnd_sat(cType m, cScType mc)

// out[i] = sat(m[i].hi16 * mc[i] << 1) >> 16  (no rounding)
#define _decl_hvx_vmpyo_s1_sat(cType, type, cScType, scType)                   \
  __attribute__((always_inline)) extern cType                                  \
      hvx_##type##_vmpyo_##type##scType##_s1_sat(cType m, cScType mc)

_decl_hvx_vmpyo_s1_rnd_sat(int, i32, int,
                           i16); // sat((m.hi16 * mc << 1) + rnd) >> 16
_decl_hvx_vmpyo_s1_sat(int, i32, int, i16); // sat(m.hi16 * mc << 1) >> 16

#undef _decl_hvx_vmpyo_s1_rnd_sat
#undef _decl_hvx_vmpyo_s1_sat

#ifdef __cplusplus
} // extern "C"
#endif

// =============================================================================
// Polymorphic wrappers
// =============================================================================

#ifndef __cplusplus

// --- C: _Generic dispatch ---

// hvx_vmpyo_s1_rnd_sat(m, mc) — odd-element multiply, <<1, rounding saturate.
#define hvx_vmpyo_s1_rnd_sat(m, mc)                                            \
  _Generic((m), int: hvx_i32_vmpyo_i32i16_s1_rnd_sat((m), (mc)))

// hvx_vmpyo_s1_sat(m, mc) — odd-element multiply, <<1, saturate (no rounding).
#define hvx_vmpyo_s1_sat(m, mc)                                                \
  _Generic((m), int: hvx_i32_vmpyo_i32i16_s1_sat((m), (mc)))

#else // __cplusplus

// --- C++: overloaded static inline functions ---

// hvx_vmpyo_s1_rnd_sat — odd-element multiply, <<1, rounding saturate.
[[gnu::always_inline, gnu::unused]] static int hvx_vmpyo_s1_rnd_sat(int m,
                                                                    int mc) {
  return hvx_i32_vmpyo_i32i16_s1_rnd_sat(m, mc);
}

// hvx_vmpyo_s1_sat — odd-element multiply, <<1, saturate (no rounding).
[[gnu::always_inline, gnu::unused]] static int hvx_vmpyo_s1_sat(int m, int mc) {
  return hvx_i32_vmpyo_i32i16_s1_sat(m, mc);
}

#endif // __cplusplus

//===-------------------------- HVX_Bfloat.h ------------------------===
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===
// Part of the Ripple vector library to support bfloat conversions
//===------------------------------------------------------------------------===

#include "lib_func_attrib.h"
#include <cmath>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <limits>
#include <ripple.h>
#include <ripple_hvx.h>

#include "HVX_Bfloat.h"

#if __has_bf16__

/// \brief extracts either the 16-bit even (oe=e) or odd (oe=o) elements of x
#define split(x, oe) Q6_Vh_vpack##oe##_VwVw(Q6_V_hi_W((x)), Q6_V_lo_W((x)))

/// \brief Bare truncation:
///  - assumes that the elements of `in` are non-NaNs.
///  - Gets a sub-optimal rounding error from just truncating
RIPPLE_INTRIN_INLINE v64bf16 ripple_ew_pure_to_bf16_trunc(v64f32 In) {
  __bf16 ForType = 0;
  return hvx_cast_from_i32(split(hvx_cast_to_i32(In), o), ForType);
}

/// \brief Truncation that's aware of NaNs.
/// Use if there is a possibility that `in`
/// contains NaNs, but if you don't care about the rounding error introduced
/// by a bare truncation.
#define shiftNan(V64I32In, b)                                                  \
  ({                                                                           \
    uint16_t ForType;                                                          \
    v64u16 Trunc = hvx_cast_from_i32(split((V64I32In), o), ForType);           \
    v64u16 CutMant = hvx_cast_from_i32(split((V64I32In), e), ForType);         \
    ripple_block_t b = ripple_set_block_shape(0, 64);                          \
    uint16_t Truncated = vec_to_ripple<64, uint16_t>(b, Trunc);                \
    uint16_t CutMantissa = vec_to_ripple<64, uint16_t>(b, CutMant);            \
    constexpr uint16_t Signless = 0x7FFF;                                      \
    constexpr uint16_t Inf = 0x7F80;                                           \
    constexpr uint16_t U16Nan = 0x7FC0;                                        \
    uint16_t Sign = 0x8000 & Truncated;                                        \
    uint16_t SignedNan = U16Nan | Sign;                                        \
    bool TruncsToInf = (Signless & Truncated) == Inf;                          \
    bool NonZeroCutMantissa = CutMantissa != 0;                                \
    uint16_t TruncWithNan =                                                    \
        (TruncsToInf & NonZeroCutMantissa) ? SignedNan : Truncated;            \
    __bf16 Result; /* bitcast to __bf16 */                                     \
    __builtin_memcpy(&Result, &TruncWithNan, sizeof(Result));                  \
    ripple_to_vec<64, __bf16>(b, Result);                                      \
  })

/// \brief Truncation that's aware of NaNs.
/// Use if there is a possibility that `in`
/// contains NaNs, but if you don't care about the rounding error introduced
/// by a bare truncation.
RIPPLE_INTRIN_INLINE v64bf16 ripple_ew_pure_to_bf16_nan(v64f32 Vin) {
  ripple_block_t b = ripple_set_block_shape(0, 64);
  return shiftNan(hvx_cast_to_i32(Vin), b);
}

/// \brief most accurate conversion from float to __bf16: even rounding,
/// NaN-aware.
RIPPLE_INTRIN_INLINE v64bf16 ripple_ew_pure_to_bf16_round(v64f32 Vin) {
  constexpr uint32_t OxeightOOO = 0x00008000u;
  constexpr uint32_t Oxonef = 0x0001FFFF;
  ripple_block_t b = ripple_set_block_shape(0, 64);
  float In = vec_to_ripple<64, float>(b, Vin);
  uint32_t bits;
  __builtin_memcpy(&bits, &In, sizeof(bits));
  // no round up if exactly .5 and even already,
  // or if it's a NaN or +/- inf
  if (((bits & Oxonef) != OxeightOOO) & ((bits & OxeightOOO) == OxeightOOO) &
      ((bits & 0x7F800000) != 0x7F800000)) {
    bits += OxeightOOO;
  }
  v64i32 PreRounded = hvx_cast_to_i32(ripple_to_vec<64, uint32_t>(b, bits));
  return shiftNan(PreRounded, b);
}

#endif // __has_bf16__

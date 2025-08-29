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

/// \brief Bare truncation:
///  - assumes that the elements of `in` are non-NaNs.
///  - Gets a sub-optimal rounding error from just truncating
RIPPLE_INTRIN_INLINE v64bf16 ripple_ew_pure_to_bf16_trunc(v64f32 In) {
  HVX_VectorPair Vin = hvx_cast_to_i32(In);
  HVX_Vector Truncated = Q6_Vh_vpacko_VwVw(Q6_V_hi_W(Vin), Q6_V_lo_W(Vin));
  __bf16 ForType = 0.0f;
  return hvx_cast_from_i32(Truncated, ForType);
}

/// \brief Truncation that's aware of NaNs.
/// Use if there is a possibility that `in`
/// contains NaNs, but if you don't care about the rounding error introduced
/// by a bare truncation.
RIPPLE_INTRIN_INLINE v64bf16 ripple_ew_pure_to_bf16_nan(v64f32 Vin) {
  ripple_block_t b = ripple_set_block_shape(0, 64);
  float In = vec_to_ripple<64, float>(b, Vin);
  __bf16 Trunc =
      vec_to_ripple<64, __bf16>(b, ripple_ew_pure_to_bf16_trunc(Vin));
  // An issue happens for NaNs whose highest mantissa bits are zero,
  // unless all of them are zero.
  __bf16 Bf16Nan = std::numeric_limits<__bf16>::quiet_NaN();
  return ripple_to_vec<64, __bf16>(b, std::isnan(In) ? Bf16Nan : Trunc);
}

/// \brief most accurate conversion from float to __bf16.
RIPPLE_INTRIN_INLINE v64bf16 ripple_ew_pure_to_bf16_round(v64f32 Vin) {
  constexpr uint16_t Oxeight = 0x8000;
  constexpr uint32_t Oxonef = 0x1FFFF;
  ripple_block_t b = ripple_set_block_shape(0, 64);
  float In = vec_to_ripple<64, float>(b, Vin);
  uint32_t bits;
  __builtin_memcpy(&bits, &In, sizeof(bits));
  // no round up if exactly .5 and even already
  if (((bits & Oxonef) != Oxeight) & ((bits & Oxeight) == Oxeight)) {
    bits += Oxeight;
  }
  constexpr uint16_t u16_silent_nan = 0x7FC0;
  uint16_t IntRes = isnan(In) ? u16_silent_nan : bits >> 16;
  __bf16 Res;
  __builtin_memcpy(&Res, &IntRes, sizeof(Res));
  return ripple_to_vec<64, __bf16>(b, Res);
}

#endif // __has_bf16__

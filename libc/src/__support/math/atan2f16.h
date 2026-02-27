//===-- Implementation header for atan2f16 ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ATAN2F16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ATAN2F16_H

#include "include/llvm-libc-macros/float16-macros.h"

#ifdef LIBC_TYPES_HAS_FLOAT16

#include "src/__support/CPP/optional.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/except_value_utils.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

namespace atan2f16_internal {

#ifndef LIBC_MATH_HAS_SKIP_ACCURATE_PASS

// (x_abs, y_abs) keys for exceptional atan2f16 inputs;
// index i -> ExceptValues input.
LIBC_INLINE_VAR constexpr size_t N_ATAN2F16_EXCEPTS = 11;
LIBC_INLINE_VAR constexpr uint16_t ATAN2F16_EXCEPT_X[] = {
    0x37DA, 0x3A01, 0x3EFD, 0x4059, 0x40CD, 0x3D84,
    0x38C2, 0x3814, 0x3596, 0x41B5, 0x3C62};
LIBC_INLINE_VAR constexpr uint16_t ATAN2F16_EXCEPT_Y[] = {
    0x3631, 0x3BFE, 0x398B, 0x3E2F, 0x4378, 0x354A,
    0x3A93, 0x3C1F, 0x4189, 0x3CA3, 0x3EB3};

// (input = index, RZ result, RU offset, RD offset, RN offset).
LIBC_INLINE_VAR constexpr fputil::ExceptValues<float16, N_ATAN2F16_EXCEPTS>
    ATAN2F16_EXCEPTS{{
        {0, 0x3957, 1, 0, 0},
        {1, 0x3B69, 1, 0, 0},
        {2, 0x360A, 1, 0, 1},
        {3, 0x38F2, 1, 0, 0},
        {4, 0x3BFE, 1, 0, 1},
        {5, 0x3387, 1, 0, 0},
        {6, 0x3B8D, 1, 0, 1},
        {7, 0x3C71, 1, 0, 1},
        {8, 0x3DC7, 1, 0, 1},
        {9, 0x362C, 1, 0, 1},
        {10, 0x3BEE, 1, 0, 0},
    }};

LIBC_INLINE constexpr cpp::optional<float16>
lookup_atan2f16_except(uint16_t x_abs, uint16_t y_abs) {
  for (size_t i = 0; i < N_ATAN2F16_EXCEPTS; ++i) {
    if (x_abs == ATAN2F16_EXCEPT_X[i] && y_abs == ATAN2F16_EXCEPT_Y[i])
      return ATAN2F16_EXCEPTS.lookup(static_cast<uint16_t>(i));
  }
  return cpp::nullopt;
}

#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

// atan(u) for u in (0, 1]: atan(u) = u * P(u^2).
LIBC_INLINE float atan_u(float u) {
  float u2 = u * u;
  float p = fputil::polyeval(u2, 0x1.fffffcp-1f, -0x1.55519ep-2f,
                             0x1.98f6a8p-3f, -0x1.1f0a92p-3f, 0x1.95b654p-4f,
                             -0x1.e65492p-5f, 0x1.8c0c36p-6f, -0x1.32316ep-8f);
  return u * p;
}

} // namespace atan2f16_internal

LIBC_INLINE float16 atan2f16(float16 y, float16 x) {
  using FPBits = fputil::FPBits<float16>;
  constexpr float PI = 0x1.921fb6p1f;
  constexpr float PI_OVER_2 = 0x1.921fb6p0f;
  constexpr float PI_OVER_4 = 0x1.921fb6p-1f;
  constexpr float THREE_PI_OVER_4 = 0x1.2d97c8p1f;
  constexpr float IS_NEG[2] = {1.0f, -1.0f};

  // const_term[x_sign][y_sign][recip]; recip = (|x| < |y|)
  constexpr float CONST_TERM[2][2][2] = {
      {{0.0f, -PI_OVER_2}, {0.0f, -PI_OVER_2}},
      {{-PI, PI_OVER_2}, {-PI, PI_OVER_2}},
  };

  FPBits x_bits(x), y_bits(y);
  bool x_sign = x_bits.sign().is_neg();
  bool y_sign = y_bits.sign().is_neg();
  x_bits.set_sign(Sign::POS);
  y_bits.set_sign(Sign::POS);
  uint16_t x_abs = x_bits.uintval();
  uint16_t y_abs = y_bits.uintval();
  uint16_t max_abs = x_abs > y_abs ? x_abs : y_abs;
  uint16_t min_abs = x_abs <= y_abs ? x_abs : y_abs;

  if (LIBC_UNLIKELY(max_abs >= 0x7c00U || min_abs == 0)) {
    if (x_bits.is_nan() || y_bits.is_nan()) {
      if (FPBits(x).is_signaling_nan() || FPBits(y).is_signaling_nan())
        fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    float xf = static_cast<float>(x);
    float yf = static_cast<float>(y);
    size_t x_except = (xf == 0.0f) ? 0 : (x_abs == 0x7c00U ? 2 : 1);
    size_t y_except = (yf == 0.0f) ? 0 : (y_abs == 0x7c00U ? 2 : 1);
    constexpr float EXCEPTS[3][3][2] = {
        {{0.0f, PI}, {0.0f, PI}, {0.0f, PI}},
        {{PI_OVER_2, PI_OVER_2}, {0.0f, 0.0f}, {0.0f, PI}},
        {{PI_OVER_2, PI_OVER_2},
         {PI_OVER_2, PI_OVER_2},
         {PI_OVER_4, THREE_PI_OVER_4}},
    };
    size_t x_neg = x_sign ? 1 : 0;
    float r = IS_NEG[y_sign] * EXCEPTS[y_except][x_except][x_neg];
    return fputil::cast<float16>(r);
  }

#ifndef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
  if (!x_sign && !y_sign) {
    if (auto r = atan2f16_internal::lookup_atan2f16_except(x_abs, y_abs);
        LIBC_UNLIKELY(r.has_value()))
      return r.value();
  }
#endif

  bool recip = x_abs < y_abs;
  float final_sign = IS_NEG[(x_sign != y_sign) != recip];
  float const_term = CONST_TERM[x_sign][y_sign][recip];

  float n = FPBits(min_abs).get_val();
  float d = FPBits(max_abs).get_val();
  float u = n / d;

  float atan_u_val = atan2f16_internal::atan_u(u);
  float r = final_sign * (const_term + atan_u_val);
  return fputil::cast<float16>(r);
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TYPES_HAS_FLOAT16

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ATAN2F16_H

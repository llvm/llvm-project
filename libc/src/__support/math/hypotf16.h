//===-- Implementation header for hypotf16 ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_HYPOTF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_HYPOTF16_H

#include "include/llvm-libc-macros/float16-macros.h"

#ifdef LIBC_TYPES_HAS_FLOAT16

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/common.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {

LIBC_INLINE float16 hypotf16(float16 x, float16 y) {
  using FPBits = fputil::FPBits<float16>;

  FPBits x_bits(x);
  FPBits y_bits(y);

  // Remove signs
  x_bits.set_sign(Sign::POS);
  y_bits.set_sign(Sign::POS);

  // Handle special cases
  // If either is NaN, return NaN
  if (LIBC_UNLIKELY(x_bits.is_nan() || y_bits.is_nan())) {
    return FPBits::quiet_nan().get_val();
  }

  // If either is infinity, return infinity
  if (LIBC_UNLIKELY(x_bits.is_inf() || y_bits.is_inf())) {
    return FPBits::inf().get_val();
  }

  // If either is zero, return the other
  if (LIBC_UNLIKELY(x_bits.is_zero())) {
    return y_bits.get_val();
  }
  if (LIBC_UNLIKELY(y_bits.is_zero())) {
    return x_bits.get_val();
  }

  // For float16, we can promote to float32 for the computation
  // to avoid overflow/underflow issues
  float x_f32 = static_cast<float>(x);
  float y_f32 = static_cast<float>(y);

  // Compute hypot using the standard formula
  float result_f32 = fputil::sqrt<float>(x_f32 * x_f32 + y_f32 * y_f32);

  // Convert back to float16
  return static_cast<float16>(result_f32);
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TYPES_HAS_FLOAT16

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_HYPOTF16_H

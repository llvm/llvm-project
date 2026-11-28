//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation for single-precision SIMD log1p.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATHVEC_LOG1PF_H
#define LLVM_LIBC_SRC___SUPPORT_MATHVEC_LOG1PF_H

#include "src/__support/CPP/simd.h"
#define LIBC_MATH_HAS_NO_ERRNO
#define LIBC_MATH_HAS_NO_EXCEPT
#define LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
#include "src/__support/math/log1pf.h"

namespace LIBC_NAMESPACE_DECL {

namespace mathvec {

namespace internal {

template <size_t N, typename Mask>
LIBC_INLINE cpp::simd<float, N> eval_small_poly(cpp::simd<float, N> x,
                                                Mask fast) {
  using Double = cpp::simd<double, N>;
  using math::double_eval::log1pf_internal::COEFFS;

  cpp::simd<float, N> poly_x = fast ? x : cpp::simd<float, N>(0.0f);
  Double xd = cpp::simd_cast<double>(poly_x);
  Double xsq = xd * xd;
  Double c0 = cpp::multiply_add(xd, Double(COEFFS[1]), Double(COEFFS[0]));
  Double c1 = cpp::multiply_add(xd, Double(COEFFS[3]), Double(COEFFS[2]));
  Double c2 = cpp::multiply_add(xd, Double(COEFFS[5]), Double(COEFFS[4]));
  Double x4 = xsq * xsq;
  Double d0 = cpp::multiply_add(xsq, c1, c0);
  Double d1 = cpp::multiply_add(xsq, Double(COEFFS[6]), c2);
  Double d2 = cpp::multiply_add(x4, d1, d0);
  return cpp::simd_cast<float>(cpp::multiply_add(xsq, d2, xd));
}

} // namespace internal

template <size_t N>
LIBC_INLINE cpp::simd<float, N> log1pf(cpp::simd<float, N> x) {
  using UInt = cpp::simd<uint32_t, N>;

  constexpr uint32_t SIGN_MASK = 0x7fff'ffffU;
  constexpr uint32_t MIN_NORMAL_BITS = 0x0080'0000U;
  constexpr uint32_t MAX_SMALL_BITS = 0x3c80'0000U;
  constexpr uint32_t RANGE_SPAN = MAX_SMALL_BITS - MIN_NORMAL_BITS;

  UInt bits = cpp::bit_cast<UInt>(x);
  UInt abs_bits = bits & SIGN_MASK;
  auto in_range = (abs_bits - MIN_NORMAL_BITS) <= RANGE_SPAN;

  if (cpp::none_of(in_range)) {
    cpp::simd<float, N> out;
    for (size_t i = 0; i < N; ++i)
      out[i] = math::log1pf(x[i]);
    return out;
  }

  auto hard = (bits == 0x3540'0003U) || (bits == 0x3710'001bU) ||
              (bits == 0xb53f'fffdU) || (bits == 0xb70f'ffe5U) ||
              (bits == 0xbb0e'c8c4U);
  auto fast = in_range && !hard;
  auto result = internal::eval_small_poly(x, fast);

  if (!cpp::all_of(fast)) {
    for (size_t i = 0; i < N; ++i) {
      if (!fast[i])
        result[i] = math::log1pf(x[i]);
    }
  }
  return result;
}

} // namespace mathvec

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATHVEC_LOG1PF_H

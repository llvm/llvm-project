//===-- Common utils for SIMD functions -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATHVEC_VECTOR_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_MATHVEC_VECTOR_UTILS_H

#include "src/__support/CPP/simd.h"
#include <tuple>

namespace LIBC_NAMESPACE_DECL {

// Casts a simd<float, N> into two simd<double, N/2>
template <size_t N>
LIBC_INLINE constexpr auto vector_float_to_double(cpp::simd<float, N> v) {
  static_assert(N % 2 == 0, "vector size must be even");
  constexpr size_t H = N / 2;

  auto parts = cpp::split<H, H>(v);
  auto lo_f = cpp::get<0>(parts);
  auto hi_f = cpp::get<1>(parts);

  auto lo_d = cpp::simd_cast<double, float, H>(lo_f);
  auto hi_d = cpp::simd_cast<double, float, H>(hi_f);

  return cpp::make_tuple(lo_d, hi_d);
}

// Casts two simd<double, N> into a simd<float, 2N>
template <size_t N>
LIBC_INLINE constexpr auto vector_double_to_float(cpp::simd<double, N> lo_d,
                                                  cpp::simd<double, N> hi_d) {

  auto lo_f = cpp::simd_cast<float, double, N>(lo_d);
  auto hi_f = cpp::simd_cast<float, double, N>(hi_d);

  return cpp::concat(lo_f, hi_f);
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATHVEC_VECTOR_UTILS_H

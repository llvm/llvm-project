//===-- Implementation header for SIMD expf ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATHVEC_EXPF_H
#define LLVM_LIBC_SRC___SUPPORT_MATHVEC_EXPF_H

#include "src/__support/CPP/simd.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/common.h"
#include "src/__support/mathvec/expf_utils.h"
#include "src/__support/mathvec/vector_utils.h"

namespace LIBC_NAMESPACE_DECL {

namespace mathvec {

template <size_t N>
LIBC_INLINE cpp::simd<double, N> inline_exp(cpp::simd<double, N> x) {
  static constexpr cpp::simd<double, N> shift = 0x1.800000000ffc0p+46;

  auto z = shift + x * 0x1.71547652b82fep+0;
  auto n = z - shift;

  auto r = x;
  r = r - n * 0x1.62e42fefa3800p-1;
  r = r - n * 0x1.ef35793c76730p-45;

  /* y = exp(r) - 1 ~= r + C0 r^2 + C1 r^3 + C2 r^4 + C3 r^5.  */
  static constexpr cpp::simd<double, N> c0 = 0x1.fffffffffdbcdp-2;
  static constexpr cpp::simd<double, N> c1 = 0x1.555555555444cp-3;
  static constexpr cpp::simd<double, N> c2 = 0x1.555573c6a9f7dp-5;
  static constexpr cpp::simd<double, N> c3 = 0x1.1111266d28935p-7;

  auto r2 = r * r;
  auto p01 = c0 + r * c1;
  auto p23 = c2 + r * c3;
  auto p04 = p01 + r2 * p23;
  auto y = r + p04 * r2;

  auto u = reinterpret_cast<cpp::simd<uint64_t, N>>(z);
  auto s = exp_lookup(u);
  return s + s * y;
}

template <size_t N>
LIBC_INLINE cpp::simd<float, N> expf(cpp::simd<float, N> x) {
  using FPBits = typename fputil::FPBits<float>;
  cpp::simd<float, N> ret;

  auto is_inf = cpp::simd_cast<bool>(x >= 0x1.62e38p+9);
  auto is_zero = cpp::simd_cast<bool>(x <= -0x1.628c2ap+9);
  auto is_special = is_inf | is_zero;

  auto special_res =
      cpp::select(is_inf, cpp::simd<float, N>(FPBits::inf().get_val()),
                  cpp::simd<float>(0.0f));

  auto [lo, hi] = vector_float_to_double(x);

  auto lo_res = inline_exp(lo);
  auto hi_res = inline_exp(hi);

  ret = vector_double_to_float(lo_res, hi_res);
  return cpp::select(is_special, special_res, ret);
}

} // namespace mathvec

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATHVEC_EXPF_H

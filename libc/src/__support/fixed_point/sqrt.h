//===-- Calculate square root of fixed point numbers. -----*- C++ -*-=========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_SQRT_H
#define LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_SQRT_H

#include "include/llvm-libc-macros/stdfix-macros.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/attributes.h"   // LIBC_INLINE
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

#include "fx_rep.h"

#ifdef LIBC_COMPILER_HAS_FIXED_POINT

namespace LIBC_NAMESPACE::fixed_point {

namespace internal {

template <typename T> struct SqrtConfig;

template <> struct SqrtConfig<unsigned short fract> {
  using Type = unsigned short fract;
  static constexpr int EXTRA_STEPS = 0;
};

template <> struct SqrtConfig<unsigned fract> {
  using Type = unsigned fract;
  static constexpr int EXTRA_STEPS = 1;
};

template <> struct SqrtConfig<unsigned long fract> {
  using Type = unsigned long fract;
  static constexpr int EXTRA_STEPS = 2;
};

template <>
struct SqrtConfig<unsigned short accum> : SqrtConfig<unsigned fract> {};

template <>
struct SqrtConfig<unsigned accum> : SqrtConfig<unsigned long fract> {};

// TODO: unsigned long accum type is 64-bit, and will need 64-bit fract type.
// Probably we will use DyadicFloat<64> for intermediate computations instead.

// Linear approximation for the initial values, with errors bounded by:
//   max(1.5 * 2^-11, eps)
// Generated with Sollya:
// > for i from 4 to 15 do {
//     P = fpminimax(sqrt(x), 1, [|8, 8|], [i * 2^-4, (i + 1)*2^-4],
//                   fixed, absolute);
//     print("{", coeff(P, 1), "uhr,", coeff(P, 0), "uhr},");
//   };
static constexpr unsigned short fract SQRT_FIRST_APPROX[12][2] = {
    {0x1.e8p-1uhr, 0x1.0cp-2uhr}, {0x1.bap-1uhr, 0x1.28p-2uhr},
    {0x1.94p-1uhr, 0x1.44p-2uhr}, {0x1.74p-1uhr, 0x1.6p-2uhr},
    {0x1.6p-1uhr, 0x1.74p-2uhr},  {0x1.4ep-1uhr, 0x1.88p-2uhr},
    {0x1.3ep-1uhr, 0x1.9cp-2uhr}, {0x1.32p-1uhr, 0x1.acp-2uhr},
    {0x1.22p-1uhr, 0x1.c4p-2uhr}, {0x1.18p-1uhr, 0x1.d4p-2uhr},
    {0x1.08p-1uhr, 0x1.fp-2uhr},  {0x1.04p-1uhr, 0x1.f8p-2uhr},
};

} // namespace internal

template <typename T>
LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_fixed_point_v<T>, T> sqrt(T x) {
  using BitType = typename FXRep<T>::StorageType;
  BitType x_bit = cpp::bit_cast<BitType>(x);

  if (LIBC_UNLIKELY(x_bit == 0))
    return FXRep<T>::ZERO();

  int leading_zeros = cpp::countl_zero(x_bit);
  constexpr int STORAGE_LENGTH = sizeof(BitType) * CHAR_BIT;
  constexpr int EXP_ADJUSTMENT = STORAGE_LENGTH - FXRep<T>::FRACTION_LEN - 1;
  // x_exp is the real exponent of the leading bit of x.
  int x_exp = EXP_ADJUSTMENT - leading_zeros;
  int shift = EXP_ADJUSTMENT - 1 - (x_exp & (~1));
  // Normalize.
  x_bit <<= shift;
  using FracType = typename internal::SqrtConfig<T>::Type;
  FracType x_frac = cpp::bit_cast<FracType>(x_bit);

  // Use use Newton method to approximate sqrt(a):
  //   x_{n + 1} = 1/2 (x_n + a / x_n)
  // For the initial values, we choose x_0

  // Use the leading 4 bits to do look up for sqrt(x).
  // After normalization, 0.25 <= x_frac < 1, so the leading 4 bits of x_frac
  // are between 0b0100 and 0b1111.  Hence the lookup table only needs 12
  // entries, and we can get the index by subtracting the leading 4 bits of
  // x_frac by 4 = 0b0100.
  int index = (x_bit >> (STORAGE_LENGTH - 4)) - 4;
  FracType a = static_cast<FracType>(internal::SQRT_FIRST_APPROX[index][0]);
  FracType b = static_cast<FracType>(internal::SQRT_FIRST_APPROX[index][1]);

  // Initial approximation step.
  // Estimated error bounds: | r - sqrt(x_frac) | < max(1.5 * 2^-11, eps).
  FracType r = a * x_frac + b;

  // Further Newton-method iterations for square-root:
  //   x_{n + 1} = 0.5 * (x_n + a / x_n)
  // We distribute and do the multiplication by 0.5 first to avoid overflow.
  // TODO: Investigate the performance and accuracy of using division-free
  // iterations from:
  //   Blanchard, J. D. and Chamberland, M., "Newton's Method Without Division",
  //   The American Mathematical Monthly (2023).
  //   https://chamberland.math.grinnell.edu/papers/newton.pdf
  for (int i = 0; i < internal::SqrtConfig<T>::EXTRA_STEPS; ++i)
    r = (r >> 1) + (x_frac >> 1) / r;

  // Re-scaling
  r >>= EXP_ADJUSTMENT - (x_exp >> 1);

  // Return result.
  return cpp::bit_cast<T>(r);
}

} // namespace LIBC_NAMESPACE::fixed_point

#endif // LIBC_COMPILER_HAS_FIXED_POINT

#endif // LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_SQRT_H

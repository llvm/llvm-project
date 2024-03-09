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
#include "src/__support/CPP/limits.h" // CHAR_BIT
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

  // Linear approximation for the initial values, with errors bounded by:
  //   max(1.5 * 2^-11, eps)
  // Generated with Sollya:
  // > for i from 4 to 15 do {
  //     P = fpminimax(sqrt(x), 1, [|8, 8|], [i * 2^-4, (i + 1)*2^-4],
  //                   fixed, absolute);
  //     print("{", coeff(P, 1), "uhr,", coeff(P, 0), "uhr},");
  //   };
  static constexpr Type FIRST_APPROX[12][2] = {
      {0x1.e8p-1uhr, 0x1.0cp-2uhr}, {0x1.bap-1uhr, 0x1.28p-2uhr},
      {0x1.94p-1uhr, 0x1.44p-2uhr}, {0x1.74p-1uhr, 0x1.6p-2uhr},
      {0x1.6p-1uhr, 0x1.74p-2uhr},  {0x1.4ep-1uhr, 0x1.88p-2uhr},
      {0x1.3ep-1uhr, 0x1.9cp-2uhr}, {0x1.32p-1uhr, 0x1.acp-2uhr},
      {0x1.22p-1uhr, 0x1.c4p-2uhr}, {0x1.18p-1uhr, 0x1.d4p-2uhr},
      {0x1.08p-1uhr, 0x1.fp-2uhr},  {0x1.04p-1uhr, 0x1.f8p-2uhr},
  };
};

template <> struct SqrtConfig<unsigned fract> {
  using Type = unsigned fract;
  static constexpr int EXTRA_STEPS = 1;

  // Linear approximation for the initial values, with errors bounded by:
  //   max(1.5 * 2^-11, eps)
  // Generated with Sollya:
  // > for i from 4 to 15 do {
  //     P = fpminimax(sqrt(x), 1, [|16, 16|], [i * 2^-4, (i + 1)*2^-4],
  //                   fixed, absolute);
  //     print("{", coeff(P, 1), "ur,", coeff(P, 0), "ur},");
  //   };
  static constexpr Type FIRST_APPROX[12][2] = {
      {0x1.e378p-1ur, 0x1.0ebp-2ur},  {0x1.b512p-1ur, 0x1.2b94p-2ur},
      {0x1.91fp-1ur, 0x1.45dcp-2ur},  {0x1.7622p-1ur, 0x1.5e24p-2ur},
      {0x1.5f5ap-1ur, 0x1.74e4p-2ur}, {0x1.4c58p-1ur, 0x1.8a4p-2ur},
      {0x1.3c1ep-1ur, 0x1.9e84p-2ur}, {0x1.2e0cp-1ur, 0x1.b1d8p-2ur},
      {0x1.21aap-1ur, 0x1.c468p-2ur}, {0x1.16bap-1ur, 0x1.d62cp-2ur},
      {0x1.0cfp-1ur, 0x1.e74cp-2ur},  {0x1.0418p-1ur, 0x1.f7ep-2ur},
  };
};

template <> struct SqrtConfig<unsigned long fract> {
  using Type = unsigned long fract;
  static constexpr int EXTRA_STEPS = 2;

  // Linear approximation for the initial values, with errors bounded by:
  //   max(1.5 * 2^-11, eps)
  // Generated with Sollya:
  // > for i from 4 to 15 do {
  //     P = fpminimax(sqrt(x), 1, [|32, 32|], [i * 2^-4, (i + 1)*2^-4],
  //                   fixed, absolute);
  //     print("{", coeff(P, 1), "ulr,", coeff(P, 0), "ulr},");
  //   };
  static constexpr Type FIRST_APPROX[12][2] = {
      {0x1.e3779b98p-1ulr, 0x1.0eaff788p-2ulr},
      {0x1.b5167872p-1ulr, 0x1.2b908ad4p-2ulr},
      {0x1.91f195cap-1ulr, 0x1.45da800cp-2ulr},
      {0x1.761ebcb4p-1ulr, 0x1.5e27004cp-2ulr},
      {0x1.5f619986p-1ulr, 0x1.74db933cp-2ulr},
      {0x1.4c583adep-1ulr, 0x1.8a3fbfccp-2ulr},
      {0x1.3c1a591cp-1ulr, 0x1.9e88373cp-2ulr},
      {0x1.2e08545ap-1ulr, 0x1.b1dd2534p-2ulr},
      {0x1.21b05c0ap-1ulr, 0x1.c45e023p-2ulr},
      {0x1.16becd02p-1ulr, 0x1.d624031p-2ulr},
      {0x1.0cf49fep-1ulr, 0x1.e743b844p-2ulr},
      {0x1.04214e9cp-1ulr, 0x1.f7ce2c3cp-2ulr},
  };
};

template <>
struct SqrtConfig<unsigned short accum> : SqrtConfig<unsigned fract> {};

template <>
struct SqrtConfig<unsigned accum> : SqrtConfig<unsigned long fract> {};

// Integer square root
template <> struct SqrtConfig<unsigned short> {
  using OutType = unsigned short accum;
  using FracType = unsigned fract;
  // For fast-but-less-accurate version
  using FastFracType = unsigned short fract;
  using HalfType = unsigned char;
};

template <> struct SqrtConfig<unsigned int> {
  using OutType = unsigned accum;
  using FracType = unsigned long fract;
  // For fast-but-less-accurate version
  using FastFracType = unsigned fract;
  using HalfType = unsigned short;
};

// TODO: unsigned long accum type is 64-bit, and will need 64-bit fract type.
// Probably we will use DyadicFloat<64> for intermediate computations instead.

} // namespace internal

// Core computation for sqrt with normalized inputs (0.25 <= x < 1).
template <typename Config>
LIBC_INLINE constexpr typename Config::Type
sqrt_core(typename Config::Type x_frac) {
  using FracType = typename Config::Type;
  using FXRep = FXRep<FracType>;
  using StorageType = typename FXRep::StorageType;
  // Exact case:
  if (x_frac == FXRep::ONE_FOURTH())
    return FXRep::ONE_HALF();

  // Use use Newton method to approximate sqrt(a):
  //   x_{n + 1} = 1/2 (x_n + a / x_n)
  // For the initial values, we choose x_0

  // Use the leading 4 bits to do look up for sqrt(x).
  // After normalization, 0.25 <= x_frac < 1, so the leading 4 bits of x_frac
  // are between 0b0100 and 0b1111.  Hence the lookup table only needs 12
  // entries, and we can get the index by subtracting the leading 4 bits of
  // x_frac by 4 = 0b0100.
  StorageType x_bit = cpp::bit_cast<StorageType>(x_frac);
  int index = (static_cast<int>(x_bit >> (FXRep::TOTAL_LEN - 4))) - 4;
  FracType a = Config::FIRST_APPROX[index][0];
  FracType b = Config::FIRST_APPROX[index][1];

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
  for (int i = 0; i < Config::EXTRA_STEPS; ++i)
    r = (r >> 1) + (x_frac >> 1) / r;

  return r;
}

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

  // Compute sqrt(x_frac) using Newton-method.
  FracType r = sqrt_core<internal::SqrtConfig<T>>(x_frac);

  // Re-scaling
  r >>= EXP_ADJUSTMENT - (x_exp >> 1);

  // Return result.
  return cpp::bit_cast<T>(r);
}

// Integer square root - Accurate version:
// Absolute errors < 2^(-fraction length).
template <typename T>
LIBC_INLINE constexpr typename internal::SqrtConfig<T>::OutType isqrt(T x) {
  using OutType = typename internal::SqrtConfig<T>::OutType;
  using FracType = typename internal::SqrtConfig<T>::FracType;

  if (x == 0)
    return FXRep<OutType>::ZERO();

  // Normalize the leading bits to the first two bits.
  // Shift and then Bit cast x to x_frac gives us:
  //   x = 2^(FRACTION_LEN + 1 - shift) * x_frac;
  int leading_zeros = cpp::countl_zero(x);
  int shift = ((leading_zeros >> 1) << 1);
  x <<= shift;
  // Convert to frac type and compute square root.
  FracType x_frac = cpp::bit_cast<FracType>(x);
  FracType r = sqrt_core<internal::SqrtConfig<FracType>>(x_frac);
  // To rescale back to the OutType (Accum)
  r >>= (shift >> 1);

  return cpp::bit_cast<OutType>(r);
}

// Integer square root - Fast but less accurate version:
// Relative errors < 2^(-fraction length).
template <typename T>
LIBC_INLINE constexpr typename internal::SqrtConfig<T>::OutType
isqrt_fast(T x) {
  using OutType = typename internal::SqrtConfig<T>::OutType;
  using FracType = typename internal::SqrtConfig<T>::FastFracType;
  using StorageType = typename FXRep<FracType>::StorageType;

  if (x == 0)
    return FXRep<OutType>::ZERO();

  // Normalize the leading bits to the first two bits.
  // Shift and then Bit cast x to x_frac gives us:
  //   x = 2^(FRACTION_LEN + 1 - shift) * x_frac;
  int leading_zeros = cpp::countl_zero(x);
  int shift = (leading_zeros & (~1));
  x <<= shift;
  // Convert to frac type and compute square root.
  FracType x_frac = cpp::bit_cast<FracType>(
      static_cast<StorageType>(x >> FXRep<FracType>::FRACTION_LEN));
  OutType r =
      static_cast<OutType>(sqrt_core<internal::SqrtConfig<FracType>>(x_frac));
  // To rescale back to the OutType (Accum)
  r <<= (FXRep<OutType>::INTEGRAL_LEN - (shift >> 1));
  return cpp::bit_cast<OutType>(r);
}

} // namespace LIBC_NAMESPACE::fixed_point

#endif // LIBC_COMPILER_HAS_FIXED_POINT

#endif // LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_SQRT_H

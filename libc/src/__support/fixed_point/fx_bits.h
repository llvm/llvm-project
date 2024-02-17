//===-- Utility class to manipulate fixed point numbers. --*- C++ -*-=========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_FXBITS_H
#define LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_FXBITS_H

#include "include/llvm-libc-macros/stdfix-macros.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/attributes.h"   // LIBC_INLINE
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

#include "fx_rep.h"

#ifdef LIBC_COMPILER_HAS_FIXED_POINT

namespace LIBC_NAMESPACE::fixed_point {

// Bit-wise operations are not available for fixed point types yet.
template <typename T>
LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_fixed_point_v<T>, T>
bit_and(T x, T y) {
  using BitType = typename FXRep<T>::StorageType;
  BitType x_bit = cpp::bit_cast<BitType>(x);
  BitType y_bit = cpp::bit_cast<BitType>(y);
  // For some reason, bit_cast cannot deduce BitType from the input.
  return cpp::bit_cast<T, BitType>(x_bit & y_bit);
}

template <typename T>
LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_fixed_point_v<T>, T>
bit_or(T x, T y) {
  using BitType = typename FXRep<T>::StorageType;
  BitType x_bit = cpp::bit_cast<BitType>(x);
  BitType y_bit = cpp::bit_cast<BitType>(y);
  // For some reason, bit_cast cannot deduce BitType from the input.
  return cpp::bit_cast<T, BitType>(x_bit | y_bit);
}

template <typename T>
LIBC_INLINE constexpr cpp::enable_if_t<cpp::is_fixed_point_v<T>, T>
bit_not(T x) {
  using BitType = typename FXRep<T>::StorageType;
  BitType x_bit = cpp::bit_cast<BitType>(x);
  // For some reason, bit_cast cannot deduce BitType from the input.
  return cpp::bit_cast<T, BitType>(~x_bit);
}

template <typename T> LIBC_INLINE constexpr T abs(T x) {
  using FXRep = FXRep<T>;
  if constexpr (FXRep::SIGN_LEN == 0)
    return x;
  else {
    if (LIBC_UNLIKELY(x == FXRep::MIN()))
      return FXRep::MAX();
    return (x < FXRep::ZERO() ? -x : x);
  }
}

// Round-to-nearest, tie-to-(+Inf)
template <typename T> LIBC_INLINE constexpr T round(T x, int n) {
  using FXRep = FXRep<T>;
  if (LIBC_UNLIKELY(n < 0))
    n = 0;
  if (LIBC_UNLIKELY(n >= FXRep::FRACTION_LEN))
    return x;

  T round_bit = FXRep::EPS() << (FXRep::FRACTION_LEN - n - 1);
  // Check for overflow.
  if (LIBC_UNLIKELY(FXRep::MAX() - round_bit < x))
    return FXRep::MAX();

  T all_ones = bit_not(FXRep::ZERO());

  int shift = FXRep::FRACTION_LEN - n;
  T rounding_mask =
      (shift == FXRep::TOTAL_LEN) ? FXRep::ZERO() : (all_ones << shift);
  return bit_and((x + round_bit), rounding_mask);
}

} // namespace LIBC_NAMESPACE::fixed_point

#endif // LIBC_COMPILER_HAS_FIXED_POINT

#endif // LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_FXBITS_H

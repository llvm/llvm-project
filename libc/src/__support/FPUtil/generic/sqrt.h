//===-- Square root of IEEE 754 floating point numbers ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_GENERIC_SQRT_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_GENERIC_SQRT_H

#include "sqrt_80_bit_long_double.h"
#include "src/__support/CPP/bit.h" // countl_zero
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/UInt128.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {
namespace fputil {

namespace internal {

template <typename T> struct SpecialLongDouble {
  static constexpr bool VALUE = false;
};

#if defined(LIBC_LONG_DOUBLE_IS_X86_FLOAT80)
template <> struct SpecialLongDouble<long double> {
  static constexpr bool VALUE = true;
};
#endif // LIBC_LONG_DOUBLE_IS_X86_FLOAT80

template <typename T>
LIBC_INLINE void normalize(int &exponent,
                           typename FPBits<T>::StorageType &mantissa) {
  const int shift = cpp::countl_zero(mantissa) -
                    (8 * sizeof(mantissa) - 1 - FPBits<T>::FRACTION_LEN);
  exponent -= shift;
  mantissa <<= shift;
}

#ifdef LIBC_LONG_DOUBLE_IS_FLOAT64
template <>
LIBC_INLINE void normalize<long double>(int &exponent, uint64_t &mantissa) {
  normalize<double>(exponent, mantissa);
}
#elif !defined(LIBC_LONG_DOUBLE_IS_X86_FLOAT80)
template <>
LIBC_INLINE void normalize<long double>(int &exponent, UInt128 &mantissa) {
  const uint64_t hi_bits = static_cast<uint64_t>(mantissa >> 64);
  const int shift =
      hi_bits ? (cpp::countl_zero(hi_bits) - 15)
              : (cpp::countl_zero(static_cast<uint64_t>(mantissa)) + 49);
  exponent -= shift;
  mantissa <<= shift;
}
#endif

} // namespace internal

// Correctly rounded IEEE 754 SQRT for all rounding modes.
// Shift-and-add algorithm.
template <typename T>
LIBC_INLINE cpp::enable_if_t<cpp::is_floating_point_v<T>, T> sqrt(T x) {

  if constexpr (internal::SpecialLongDouble<T>::VALUE) {
    // Special 80-bit long double.
    return x86::sqrt(x);
  } else {
    // IEEE floating points formats.
    using FPBits_t = typename fputil::FPBits<T>;
    using StorageType = typename FPBits_t::StorageType;
    constexpr StorageType ONE = StorageType(1) << FPBits_t::FRACTION_LEN;
    constexpr auto FLT_NAN = FPBits_t::quiet_nan().get_val();

    FPBits_t bits(x);

    if (bits == FPBits_t::inf(Sign::POS) || bits.is_zero() || bits.is_nan()) {
      // sqrt(+Inf) = +Inf
      // sqrt(+0) = +0
      // sqrt(-0) = -0
      // sqrt(NaN) = NaN
      // sqrt(-NaN) = -NaN
      return x;
    } else if (bits.is_neg()) {
      // sqrt(-Inf) = NaN
      // sqrt(-x) = NaN
      return FLT_NAN;
    } else {
      int x_exp = bits.get_exponent();
      StorageType x_mant = bits.get_mantissa();

      // Step 1a: Normalize denormal input and append hidden bit to the mantissa
      if (bits.is_subnormal()) {
        ++x_exp; // let x_exp be the correct exponent of ONE bit.
        internal::normalize<T>(x_exp, x_mant);
      } else {
        x_mant |= ONE;
      }

      // Step 1b: Make sure the exponent is even.
      if (x_exp & 1) {
        --x_exp;
        x_mant <<= 1;
      }

      // After step 1b, x = 2^(x_exp) * x_mant, where x_exp is even, and
      // 1 <= x_mant < 4.  So sqrt(x) = 2^(x_exp / 2) * y, with 1 <= y < 2.
      // Notice that the output of sqrt is always in the normal range.
      // To perform shift-and-add algorithm to find y, let denote:
      //   y(n) = 1.y_1 y_2 ... y_n, we can define the nth residue to be:
      //   r(n) = 2^n ( x_mant - y(n)^2 ).
      // That leads to the following recurrence formula:
      //   r(n) = 2*r(n-1) - y_n*[ 2*y(n-1) + 2^(-n-1) ]
      // with the initial conditions: y(0) = 1, and r(0) = x - 1.
      // So the nth digit y_n of the mantissa of sqrt(x) can be found by:
      //   y_n = 1 if 2*r(n-1) >= 2*y(n - 1) + 2^(-n-1)
      //         0 otherwise.
      StorageType y = ONE;
      StorageType r = x_mant - ONE;

      for (StorageType current_bit = ONE >> 1; current_bit; current_bit >>= 1) {
        r <<= 1;
        StorageType tmp = (y << 1) + current_bit; // 2*y(n - 1) + 2^(-n-1)
        if (r >= tmp) {
          r -= tmp;
          y += current_bit;
        }
      }

      // We compute one more iteration in order to round correctly.
      bool lsb = static_cast<bool>(y & 1); // Least significant bit
      bool rb = false;                     // Round bit
      r <<= 2;
      StorageType tmp = (y << 2) + 1;
      if (r >= tmp) {
        r -= tmp;
        rb = true;
      }

      // Remove hidden bit and append the exponent field.
      x_exp = ((x_exp >> 1) + FPBits_t::EXP_BIAS);

      y = (y - ONE) |
          (static_cast<StorageType>(x_exp) << FPBits_t::FRACTION_LEN);

      switch (quick_get_round()) {
      case FE_TONEAREST:
        // Round to nearest, ties to even
        if (rb && (lsb || (r != 0)))
          ++y;
        break;
      case FE_UPWARD:
        if (rb || (r != 0))
          ++y;
        break;
      }

      return cpp::bit_cast<T>(y);
    }
  }
}

} // namespace fputil
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_GENERIC_SQRT_H

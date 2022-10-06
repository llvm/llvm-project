//===-- Square root of IEEE 754 floating point numbers ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_GENERIC_SQRT_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_GENERIC_SQRT_H

#include "sqrt_80_bit_long_double.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PlatformDefs.h"
#include "src/__support/UInt128.h"
#include "src/__support/builtin_wrappers.h"

namespace __llvm_libc {
namespace fputil {

namespace internal {

template <typename T> struct SpecialLongDouble {
  static constexpr bool VALUE = false;
};

#if defined(SPECIAL_X86_LONG_DOUBLE)
template <> struct SpecialLongDouble<long double> {
  static constexpr bool VALUE = true;
};
#endif // SPECIAL_X86_LONG_DOUBLE

template <typename T>
static inline void normalize(int &exponent,
                             typename FPBits<T>::UIntType &mantissa) {
  const int shift = unsafe_clz(mantissa) -
                    (8 * sizeof(mantissa) - 1 - MantissaWidth<T>::VALUE);
  exponent -= shift;
  mantissa <<= shift;
}

#ifdef LONG_DOUBLE_IS_DOUBLE
template <>
inline void normalize<long double>(int &exponent, uint64_t &mantissa) {
  normalize<double>(exponent, mantissa);
}
#elif !defined(SPECIAL_X86_LONG_DOUBLE)
template <>
inline void normalize<long double>(int &exponent, UInt128 &mantissa) {
  const uint64_t hi_bits = static_cast<uint64_t>(mantissa >> 64);
  const int shift = hi_bits
                        ? (unsafe_clz(hi_bits) - 15)
                        : (unsafe_clz(static_cast<uint64_t>(mantissa)) + 49);
  exponent -= shift;
  mantissa <<= shift;
}
#endif

} // namespace internal

// Correctly rounded IEEE 754 SQRT for all rounding modes.
// Shift-and-add algorithm.
template <typename T>
static inline cpp::enable_if_t<cpp::is_floating_point_v<T>, T> sqrt(T x) {

  if constexpr (internal::SpecialLongDouble<T>::VALUE) {
    // Special 80-bit long double.
    return x86::sqrt(x);
  } else {
    // IEEE floating points formats.
    using UIntType = typename FPBits<T>::UIntType;
    constexpr UIntType ONE = UIntType(1) << MantissaWidth<T>::VALUE;

    FPBits<T> bits(x);

    if (bits.is_inf_or_nan()) {
      if (bits.get_sign() && (bits.get_mantissa() == 0)) {
        // sqrt(-Inf) = NaN
        return FPBits<T>::build_nan(ONE >> 1);
      } else {
        // sqrt(NaN) = NaN
        // sqrt(+Inf) = +Inf
        return x;
      }
    } else if (bits.is_zero()) {
      // sqrt(+0) = +0
      // sqrt(-0) = -0
      return x;
    } else if (bits.get_sign()) {
      // sqrt( negative numbers ) = NaN
      return FPBits<T>::build_nan(ONE >> 1);
    } else {
      int x_exp = bits.get_exponent();
      UIntType x_mant = bits.get_mantissa();

      // Step 1a: Normalize denormal input and append hidden bit to the mantissa
      if (bits.get_unbiased_exponent() == 0) {
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
      UIntType y = ONE;
      UIntType r = x_mant - ONE;

      for (UIntType current_bit = ONE >> 1; current_bit; current_bit >>= 1) {
        r <<= 1;
        UIntType tmp = (y << 1) + current_bit; // 2*y(n - 1) + 2^(-n-1)
        if (r >= tmp) {
          r -= tmp;
          y += current_bit;
        }
      }

      // We compute one more iteration in order to round correctly.
      bool lsb = y & 1; // Least significant bit
      bool rb = false;  // Round bit
      r <<= 2;
      UIntType tmp = (y << 2) + 1;
      if (r >= tmp) {
        r -= tmp;
        rb = true;
      }

      // Remove hidden bit and append the exponent field.
      x_exp = ((x_exp >> 1) + FPBits<T>::EXPONENT_BIAS);

      y = (y - ONE) | (static_cast<UIntType>(x_exp) << MantissaWidth<T>::VALUE);

      switch (get_round()) {
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
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_GENERIC_SQRT_H

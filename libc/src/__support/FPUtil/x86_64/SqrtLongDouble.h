//===-- Square root of x86 long double numbers ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_SQRT_LONG_DOUBLE_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_SQRT_LONG_DOUBLE_H

#include "src/__support/architectures.h"

#if !defined(LLVM_LIBC_ARCH_X86)
#error "Invalid include"
#endif

#include "src/__support/CPP/TypeTraits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/Sqrt.h"

namespace __llvm_libc {
namespace fputil {

namespace internal {

template <>
inline void normalize<long double>(int &exponent, __uint128_t &mantissa) {
  // Use binary search to shift the leading 1 bit similar to float.
  // With MantissaWidth<long double> = 63, it will take
  // ceil(log2(63)) = 6 steps checking the mantissa bits.
  constexpr int NSTEPS = 6; // = ceil(log2(MantissaWidth))
  constexpr __uint128_t BOUNDS[NSTEPS] = {
      __uint128_t(1) << 32, __uint128_t(1) << 48, __uint128_t(1) << 56,
      __uint128_t(1) << 60, __uint128_t(1) << 62, __uint128_t(1) << 63};
  constexpr int SHIFTS[NSTEPS] = {32, 16, 8, 4, 2, 1};

  for (int i = 0; i < NSTEPS; ++i) {
    if (mantissa < BOUNDS[i]) {
      exponent -= SHIFTS[i];
      mantissa <<= SHIFTS[i];
    }
  }
}

} // namespace internal

// Correctly rounded SQRT with round to nearest, ties to even.
// Shift-and-add algorithm.
template <> inline long double sqrt<long double, 0>(long double x) {
  using UIntType = typename FPBits<long double>::UIntType;
  constexpr UIntType ONE = UIntType(1)
                           << int(MantissaWidth<long double>::VALUE);

  FPBits<long double> bits(x);

  if (bits.is_inf_or_nan()) {
    if (bits.get_sign() && (bits.get_mantissa() == 0)) {
      // sqrt(-Inf) = NaN
      return FPBits<long double>::build_nan(ONE >> 1);
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
    return FPBits<long double>::build_nan(ONE >> 1);
  } else {
    int x_exp = bits.get_exponent();
    UIntType x_mant = bits.get_mantissa();

    // Step 1a: Normalize denormal input
    if (bits.get_implicit_bit()) {
      x_mant |= ONE;
    } else if (bits.get_unbiased_exponent() == 0) {
      internal::normalize<long double>(x_exp, x_mant);
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

    // Append the exponent field.
    x_exp = ((x_exp >> 1) + FPBits<long double>::EXPONENT_BIAS);
    y |= (static_cast<UIntType>(x_exp)
          << (MantissaWidth<long double>::VALUE + 1));

    // Round to nearest, ties to even
    if (rb && (lsb || (r != 0))) {
      ++y;
    }

    // Extract output
    FPBits<long double> out(0.0L);
    out.set_unbiased_exponent(x_exp);
    out.set_implicit_bit(1);
    out.set_mantissa((y & (ONE - 1)));

    return out;
  }
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_SQRT_LONG_DOUBLE_H

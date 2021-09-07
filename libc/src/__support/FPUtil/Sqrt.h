//===-- Square root of IEEE 754 floating point numbers ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_SQRT_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_SQRT_H

#include "FPBits.h"
#include "PlatformDefs.h"

#include "utils/CPP/TypeTraits.h"

namespace __llvm_libc {
namespace fputil {

namespace internal {

template <typename T>
static inline void normalize(int &exponent,
                             typename FPBits<T>::UIntType &mantissa);

template <> inline void normalize<float>(int &exponent, uint32_t &mantissa) {
  // Use binary search to shift the leading 1 bit.
  // With MantissaWidth<float> = 23, it will take
  // ceil(log2(23)) = 5 steps checking the mantissa bits as followed:
  // Step 1: 0000 0000 0000 XXXX XXXX XXXX
  // Step 2: 0000 00XX XXXX XXXX XXXX XXXX
  // Step 3: 000X XXXX XXXX XXXX XXXX XXXX
  // Step 4: 00XX XXXX XXXX XXXX XXXX XXXX
  // Step 5: 0XXX XXXX XXXX XXXX XXXX XXXX
  constexpr int nsteps = 5; // = ceil(log2(MantissaWidth))
  constexpr uint32_t bounds[nsteps] = {1 << 12, 1 << 18, 1 << 21, 1 << 22,
                                       1 << 23};
  constexpr int shifts[nsteps] = {12, 6, 3, 2, 1};

  for (int i = 0; i < nsteps; ++i) {
    if (mantissa < bounds[i]) {
      exponent -= shifts[i];
      mantissa <<= shifts[i];
    }
  }
}

template <> inline void normalize<double>(int &exponent, uint64_t &mantissa) {
  // Use binary search to shift the leading 1 bit similar to float.
  // With MantissaWidth<double> = 52, it will take
  // ceil(log2(52)) = 6 steps checking the mantissa bits.
  constexpr int nsteps = 6; // = ceil(log2(MantissaWidth))
  constexpr uint64_t bounds[nsteps] = {1ULL << 26, 1ULL << 39, 1ULL << 46,
                                       1ULL << 49, 1ULL << 51, 1ULL << 52};
  constexpr int shifts[nsteps] = {27, 14, 7, 4, 2, 1};

  for (int i = 0; i < nsteps; ++i) {
    if (mantissa < bounds[i]) {
      exponent -= shifts[i];
      mantissa <<= shifts[i];
    }
  }
}

#ifdef LONG_DOUBLE_IS_DOUBLE
template <>
inline void normalize<long double>(int &exponent, uint64_t &mantissa) {
  normalize<double>(exponent, mantissa);
}
#elif !defined(SPECIAL_X86_LONG_DOUBLE)
template <>
inline void normalize<long double>(int &exponent, __uint128_t &mantissa) {
  // Use binary search to shift the leading 1 bit similar to float.
  // With MantissaWidth<long double> = 112, it will take
  // ceil(log2(112)) = 7 steps checking the mantissa bits.
  constexpr int nsteps = 7; // = ceil(log2(MantissaWidth))
  constexpr __uint128_t bounds[nsteps] = {
      __uint128_t(1) << 56,  __uint128_t(1) << 84,  __uint128_t(1) << 98,
      __uint128_t(1) << 105, __uint128_t(1) << 109, __uint128_t(1) << 111,
      __uint128_t(1) << 112};
  constexpr int shifts[nsteps] = {57, 29, 15, 8, 4, 2, 1};

  for (int i = 0; i < nsteps; ++i) {
    if (mantissa < bounds[i]) {
      exponent -= shifts[i];
      mantissa <<= shifts[i];
    }
  }
}
#endif

} // namespace internal

// Correctly rounded IEEE 754 SQRT with round to nearest, ties to even.
// Shift-and-add algorithm.
template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T sqrt(T x) {
  using UIntType = typename FPBits<T>::UIntType;
  constexpr UIntType One = UIntType(1) << MantissaWidth<T>::value;

  FPBits<T> bits(x);

  if (bits.isInfOrNaN()) {
    if (bits.getSign() && (bits.getMantissa() == 0)) {
      // sqrt(-Inf) = NaN
      return FPBits<T>::buildNaN(One >> 1);
    } else {
      // sqrt(NaN) = NaN
      // sqrt(+Inf) = +Inf
      return x;
    }
  } else if (bits.isZero()) {
    // sqrt(+0) = +0
    // sqrt(-0) = -0
    return x;
  } else if (bits.getSign()) {
    // sqrt( negative numbers ) = NaN
    return FPBits<T>::buildNaN(One >> 1);
  } else {
    int xExp = bits.getExponent();
    UIntType xMant = bits.getMantissa();

    // Step 1a: Normalize denormal input and append hidden bit to the mantissa
    if (bits.getUnbiasedExponent() == 0) {
      ++xExp; // let xExp be the correct exponent of One bit.
      internal::normalize<T>(xExp, xMant);
    } else {
      xMant |= One;
    }

    // Step 1b: Make sure the exponent is even.
    if (xExp & 1) {
      --xExp;
      xMant <<= 1;
    }

    // After step 1b, x = 2^(xExp) * xMant, where xExp is even, and
    // 1 <= xMant < 4.  So sqrt(x) = 2^(xExp / 2) * y, with 1 <= y < 2.
    // Notice that the output of sqrt is always in the normal range.
    // To perform shift-and-add algorithm to find y, let denote:
    //   y(n) = 1.y_1 y_2 ... y_n, we can define the nth residue to be:
    //   r(n) = 2^n ( xMant - y(n)^2 ).
    // That leads to the following recurrence formula:
    //   r(n) = 2*r(n-1) - y_n*[ 2*y(n-1) + 2^(-n-1) ]
    // with the initial conditions: y(0) = 1, and r(0) = x - 1.
    // So the nth digit y_n of the mantissa of sqrt(x) can be found by:
    //   y_n = 1 if 2*r(n-1) >= 2*y(n - 1) + 2^(-n-1)
    //         0 otherwise.
    UIntType y = One;
    UIntType r = xMant - One;

    for (UIntType current_bit = One >> 1; current_bit; current_bit >>= 1) {
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
    xExp = ((xExp >> 1) + FPBits<T>::exponentBias);

    y = (y - One) | (static_cast<UIntType>(xExp) << MantissaWidth<T>::value);
    // Round to nearest, ties to even
    if (rb && (lsb || (r != 0))) {
      ++y;
    }

    return *reinterpret_cast<T *>(&y);
  }
}

} // namespace fputil
} // namespace __llvm_libc

#ifdef SPECIAL_X86_LONG_DOUBLE
#include "SqrtLongDoubleX86.h"
#endif // SPECIAL_X86_LONG_DOUBLE

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_SQRT_H

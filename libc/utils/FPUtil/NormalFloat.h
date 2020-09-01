//===-- A class to store a normalized floating point number -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_NORMAL_FLOAT_H
#define LLVM_LIBC_UTILS_FPUTIL_NORMAL_FLOAT_H

#include "FPBits.h"

#include "utils/CPP/TypeTraits.h"

#include <stdint.h>

namespace __llvm_libc {
namespace fputil {

// A class which stores the normalized form of a floating point value.
// The special IEEE-754 bits patterns of Zero, infinity and NaNs are
// are not handled by this class.
//
// A normalized floating point number is of this form:
//    (-1)*sign * 2^exponent * <mantissa>
// where <mantissa> is of the form 1.<...>.
template <typename T> struct NormalFloat {
  static_assert(
      cpp::IsFloatingPointType<T>::Value,
      "NormalFloat template parameter has to be a floating point type.");

  using UIntType = typename FPBits<T>::UIntType;
  static constexpr UIntType one = (UIntType(1) << MantissaWidth<T>::value);

  // Unbiased exponent value.
  int32_t exponent;

  UIntType mantissa;
  // We want |UIntType| to have atleast one bit more than the actual mantissa
  // bit width to accommodate the implicit 1 value.
  static_assert(sizeof(UIntType) * 8 >= MantissaWidth<T>::value + 1,
                "Bad type for mantissa in NormalFloat.");

  bool sign;

  NormalFloat(int32_t e, UIntType m, bool s)
      : exponent(e), mantissa(m), sign(s) {
    if (mantissa >= one)
      return;

    unsigned normalizationShift = evaluateNormalizationShift(mantissa);
    mantissa = mantissa << normalizationShift;
    exponent -= normalizationShift;
  }

  explicit NormalFloat(T x) { initFromBits(FPBits<T>(x)); }

  explicit NormalFloat(FPBits<T> bits) { initFromBits(bits); }

  // Compares this normalized number with another normalized number.
  // Returns -1 is this number is less than |other|, 0 if this number is equal
  // to |other|, and 1 if this number is greater than |other|.
  int cmp(const NormalFloat<T> &other) const {
    if (sign != other.sign)
      return sign ? -1 : 1;

    if (exponent > other.exponent) {
      return sign ? -1 : 1;
    } else if (exponent == other.exponent) {
      if (mantissa > other.mantissa)
        return sign ? -1 : 1;
      else if (mantissa == other.mantissa)
        return 0;
      else
        return sign ? 1 : -1;
    } else {
      return sign ? 1 : -1;
    }
  }

  // Returns a new normalized floating point number which is equal in value
  // to this number multiplied by 2^e. That is:
  //     new = this *  2^e
  NormalFloat<T> mul2(int e) const {
    NormalFloat<T> result = *this;
    result.exponent += e;
    return result;
  }

  operator T() const {
    int biasedExponent = exponent + FPBits<T>::exponentBias;
    // Max exponent is of the form 0xFF...E. That is why -2 and not -1.
    constexpr int maxExponentValue = (1 << ExponentWidth<T>::value) - 2;
    if (biasedExponent > maxExponentValue) {
      // TODO: Should infinity with the correct sign be returned?
      return FPBits<T>::buildNaN(1);
    }

    FPBits<T> result(T(0.0));

    constexpr int subnormalExponent = -FPBits<T>::exponentBias + 1;
    if (exponent < subnormalExponent) {
      unsigned shift = subnormalExponent - exponent;
      if (shift <= MantissaWidth<T>::value) {
        // Generate a subnormal number. Might lead to loss of precision.
        result.exponent = 0;
        result.mantissa = mantissa >> shift;
        result.sign = sign;
        return result;
      } else {
        // TODO: Should zero with the correct sign be returned?
        return FPBits<T>::buildNaN(1);
      }
    }

    result.exponent = exponent + FPBits<T>::exponentBias;
    result.mantissa = mantissa;
    result.sign = sign;
    return result;
  }

private:
  void initFromBits(FPBits<T> bits) {
    sign = bits.sign;

    if (bits.isInfOrNaN() || bits.isZero()) {
      // Ignore special bit patterns. Implementations deal with them separately
      // anyway so this should not be a problem.
      exponent = 0;
      mantissa = 0;
      return;
    }

    // Normalize subnormal numbers.
    if (bits.exponent == 0) {
      unsigned shift = evaluateNormalizationShift(bits.mantissa);
      mantissa = UIntType(bits.mantissa) << shift;
      exponent = 1 - FPBits<T>::exponentBias - shift;
    } else {
      exponent = bits.exponent - FPBits<T>::exponentBias;
      mantissa = one | bits.mantissa;
    }
  }

  unsigned evaluateNormalizationShift(UIntType m) {
    unsigned shift = 0;
    for (; (one & m) == 0 && (shift < MantissaWidth<T>::value);
         m <<= 1, ++shift)
      ;
    return shift;
  }
};

#if defined(__x86_64__) || defined(__i386__)
template <>
inline void NormalFloat<long double>::initFromBits(FPBits<long double> bits) {
  sign = bits.sign;

  if (bits.isInfOrNaN() || bits.isZero()) {
    // Ignore special bit patterns. Implementations deal with them separately
    // anyway so this should not be a problem.
    exponent = 0;
    mantissa = 0;
    return;
  }

  if (bits.exponent == 0) {
    if (bits.implicitBit == 0) {
      // Since we ignore zero value, the mantissa in this case is non-zero.
      int normalizationShift = evaluateNormalizationShift(bits.mantissa);
      exponent = -16382 - normalizationShift;
      mantissa = (bits.mantissa << normalizationShift);
    } else {
      exponent = -16382;
      mantissa = one | bits.mantissa;
    }
  } else {
    if (bits.implicitBit == 0) {
      // Invalid number so just store 0 similar to a NaN.
      exponent = 0;
      mantissa = 0;
    } else {
      exponent = bits.exponent - 16383;
      mantissa = one | bits.mantissa;
    }
  }
}

template <> inline NormalFloat<long double>::operator long double() const {
  int biasedExponent = exponent + FPBits<long double>::exponentBias;
  // Max exponent is of the form 0xFF...E. That is why -2 and not -1.
  constexpr int maxExponentValue = (1 << ExponentWidth<long double>::value) - 2;
  if (biasedExponent > maxExponentValue) {
    // TODO: Should infinity with the correct sign be returned?
    return FPBits<long double>::buildNaN(1);
  }

  FPBits<long double> result(0.0l);

  constexpr int subnormalExponent = -FPBits<long double>::exponentBias + 1;
  if (exponent < subnormalExponent) {
    unsigned shift = subnormalExponent - exponent;
    if (shift <= MantissaWidth<long double>::value) {
      // Generate a subnormal number. Might lead to loss of precision.
      result.exponent = 0;
      result.mantissa = mantissa >> shift;
      result.implicitBit = 0;
      result.sign = sign;
      return result;
    } else {
      // TODO: Should zero with the correct sign be returned?
      return FPBits<long double>::buildNaN(1);
    }
  }

  result.exponent = biasedExponent;
  result.mantissa = mantissa;
  result.implicitBit = 1;
  result.sign = sign;
  return result;
}
#endif

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_NORMAL_FLOAT_H

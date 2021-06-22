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
      return sign ? T(FPBits<T>::negInf()) : T(FPBits<T>::inf());
    }

    FPBits<T> result(T(0.0));
    result.encoding.sign = sign;

    constexpr int subnormalExponent = -FPBits<T>::exponentBias + 1;
    if (exponent < subnormalExponent) {
      unsigned shift = subnormalExponent - exponent;
      // Since exponent > subnormalExponent, shift is strictly greater than
      // zero.
      if (shift <= MantissaWidth<T>::value + 1) {
        // Generate a subnormal number. Might lead to loss of precision.
        // We round to nearest and round halfway cases to even.
        const UIntType shiftOutMask = (UIntType(1) << shift) - 1;
        const UIntType shiftOutValue = mantissa & shiftOutMask;
        const UIntType halfwayValue = UIntType(1) << (shift - 1);
        result.encoding.exponent = 0;
        result.encoding.mantissa = mantissa >> shift;
        UIntType newMantissa = result.encoding.mantissa;
        if (shiftOutValue > halfwayValue) {
          newMantissa += 1;
        } else if (shiftOutValue == halfwayValue) {
          // Round to even.
          if (result.encoding.mantissa & 0x1)
            newMantissa += 1;
        }
        result.encoding.mantissa = newMantissa;
        // Adding 1 to mantissa can lead to overflow. This can only happen if
        // mantissa was all ones (0b111..11). For such a case, we will carry
        // the overflow into the exponent.
        if (newMantissa == one)
          result.encoding.exponent = 1;
        return T(result);
      } else {
        return T(result);
      }
    }

    result.encoding.exponent = exponent + FPBits<T>::exponentBias;
    result.encoding.mantissa = mantissa;
    return T(result);
  }

private:
  void initFromBits(FPBits<T> bits) {
    sign = bits.encoding.sign;

    if (bits.isInfOrNaN() || bits.isZero()) {
      // Ignore special bit patterns. Implementations deal with them separately
      // anyway so this should not be a problem.
      exponent = 0;
      mantissa = 0;
      return;
    }

    // Normalize subnormal numbers.
    if (bits.encoding.exponent == 0) {
      unsigned shift = evaluateNormalizationShift(bits.encoding.mantissa);
      mantissa = UIntType(bits.encoding.mantissa) << shift;
      exponent = 1 - FPBits<T>::exponentBias - shift;
    } else {
      exponent = bits.encoding.exponent - FPBits<T>::exponentBias;
      mantissa = one | bits.encoding.mantissa;
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

#ifdef SPECIAL_X86_LONG_DOUBLE
template <>
inline void NormalFloat<long double>::initFromBits(FPBits<long double> bits) {
  sign = bits.encoding.sign;

  if (bits.isInfOrNaN() || bits.isZero()) {
    // Ignore special bit patterns. Implementations deal with them separately
    // anyway so this should not be a problem.
    exponent = 0;
    mantissa = 0;
    return;
  }

  if (bits.encoding.exponent == 0) {
    if (bits.encoding.implicitBit == 0) {
      // Since we ignore zero value, the mantissa in this case is non-zero.
      int normalizationShift =
          evaluateNormalizationShift(bits.encoding.mantissa);
      exponent = -16382 - normalizationShift;
      mantissa = (bits.encoding.mantissa << normalizationShift);
    } else {
      exponent = -16382;
      mantissa = one | bits.encoding.mantissa;
    }
  } else {
    if (bits.encoding.implicitBit == 0) {
      // Invalid number so just store 0 similar to a NaN.
      exponent = 0;
      mantissa = 0;
    } else {
      exponent = bits.encoding.exponent - 16383;
      mantissa = one | bits.encoding.mantissa;
    }
  }
}

template <> inline NormalFloat<long double>::operator long double() const {
  int biasedExponent = exponent + FPBits<long double>::exponentBias;
  // Max exponent is of the form 0xFF...E. That is why -2 and not -1.
  constexpr int maxExponentValue = (1 << ExponentWidth<long double>::value) - 2;
  if (biasedExponent > maxExponentValue) {
    return sign ? FPBits<long double>::negInf() : FPBits<long double>::inf();
  }

  FPBits<long double> result(0.0l);
  result.encoding.sign = sign;

  constexpr int subnormalExponent = -FPBits<long double>::exponentBias + 1;
  if (exponent < subnormalExponent) {
    unsigned shift = subnormalExponent - exponent;
    if (shift <= MantissaWidth<long double>::value + 1) {
      // Generate a subnormal number. Might lead to loss of precision.
      // We round to nearest and round halfway cases to even.
      const UIntType shiftOutMask = (UIntType(1) << shift) - 1;
      const UIntType shiftOutValue = mantissa & shiftOutMask;
      const UIntType halfwayValue = UIntType(1) << (shift - 1);
      result.encoding.exponent = 0;
      result.encoding.mantissa = mantissa >> shift;
      UIntType newMantissa = result.encoding.mantissa;
      if (shiftOutValue > halfwayValue) {
        newMantissa += 1;
      } else if (shiftOutValue == halfwayValue) {
        // Round to even.
        if (result.encoding.mantissa & 0x1)
          newMantissa += 1;
      }
      result.encoding.mantissa = newMantissa;
      // Adding 1 to mantissa can lead to overflow. This can only happen if
      // mantissa was all ones (0b111..11). For such a case, we will carry
      // the overflow into the exponent and set the implicit bit to 1.
      if (newMantissa == one) {
        result.encoding.exponent = 1;
        result.encoding.implicitBit = 1;
      } else {
        result.encoding.implicitBit = 0;
      }
      return static_cast<long double>(result);
    } else {
      return static_cast<long double>(result);
    }
  }

  result.encoding.exponent = biasedExponent;
  result.encoding.mantissa = mantissa;
  result.encoding.implicitBit = 1;
  return static_cast<long double>(result);
}
#endif // SPECIAL_X86_LONG_DOUBLE

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_NORMAL_FLOAT_H

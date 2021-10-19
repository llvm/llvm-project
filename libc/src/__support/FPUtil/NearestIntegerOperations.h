//===-- Nearest integer floating-point operations ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_NEAREST_INTEGER_OPERATIONS_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_NEAREST_INTEGER_OPERATIONS_H

#include "FEnvUtils.h"
#include "FPBits.h"

#include "utils/CPP/TypeTraits.h"

#include <math.h>
#if math_errhandling & MATH_ERRNO
#include <errno.h>
#endif

namespace __llvm_libc {
namespace fputil {

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T trunc(T x) {
  FPBits<T> bits(x);

  // If x is infinity or NaN, return it.
  // If it is zero also we should return it as is, but the logic
  // later in this function takes care of it. But not doing a zero
  // check, we improve the run time of non-zero values.
  if (bits.isInfOrNaN())
    return x;

  int exponent = bits.getExponent();

  // If the exponent is greater than the most negative mantissa
  // exponent, then x is already an integer.
  if (exponent >= static_cast<int>(MantissaWidth<T>::value))
    return x;

  // If the exponent is such that abs(x) is less than 1, then return 0.
  if (exponent <= -1) {
    if (bits.getSign())
      return T(-0.0);
    else
      return T(0.0);
  }

  int trimSize = MantissaWidth<T>::value - exponent;
  bits.setMantissa((bits.getMantissa() >> trimSize) << trimSize);
  return T(bits);
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T ceil(T x) {
  FPBits<T> bits(x);

  // If x is infinity NaN or zero, return it.
  if (bits.isInfOrNaN() || bits.isZero())
    return x;

  bool isNeg = bits.getSign();
  int exponent = bits.getExponent();

  // If the exponent is greater than the most negative mantissa
  // exponent, then x is already an integer.
  if (exponent >= static_cast<int>(MantissaWidth<T>::value))
    return x;

  if (exponent <= -1) {
    if (isNeg)
      return T(-0.0);
    else
      return T(1.0);
  }

  uint32_t trimSize = MantissaWidth<T>::value - exponent;
  bits.setMantissa((bits.getMantissa() >> trimSize) << trimSize);
  T truncValue = T(bits);

  // If x is already an integer, return it.
  if (truncValue == x)
    return x;

  // If x is negative, the ceil operation is equivalent to the trunc operation.
  if (isNeg)
    return truncValue;

  return truncValue + T(1.0);
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T floor(T x) {
  FPBits<T> bits(x);
  if (bits.getSign()) {
    return -ceil(-x);
  } else {
    return trunc(x);
  }
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T round(T x) {
  using UIntType = typename FPBits<T>::UIntType;
  FPBits<T> bits(x);

  // If x is infinity NaN or zero, return it.
  if (bits.isInfOrNaN() || bits.isZero())
    return x;

  bool isNeg = bits.getSign();
  int exponent = bits.getExponent();

  // If the exponent is greater than the most negative mantissa
  // exponent, then x is already an integer.
  if (exponent >= static_cast<int>(MantissaWidth<T>::value))
    return x;

  if (exponent == -1) {
    // Absolute value of x is greater than equal to 0.5 but less than 1.
    if (isNeg)
      return T(-1.0);
    else
      return T(1.0);
  }

  if (exponent <= -2) {
    // Absolute value of x is less than 0.5.
    if (isNeg)
      return T(-0.0);
    else
      return T(0.0);
  }

  uint32_t trimSize = MantissaWidth<T>::value - exponent;
  bool halfBitSet = bits.getMantissa() & (UIntType(1) << (trimSize - 1));
  bits.setMantissa((bits.getMantissa() >> trimSize) << trimSize);
  T truncValue = T(bits);

  // If x is already an integer, return it.
  if (truncValue == x)
    return x;

  if (!halfBitSet) {
    // Franctional part is less than 0.5 so round value is the
    // same as the trunc value.
    return truncValue;
  } else {
    return isNeg ? truncValue - T(1.0) : truncValue + T(1.0);
  }
}

template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T roundUsingCurrentRoundingMode(T x) {
  using UIntType = typename FPBits<T>::UIntType;
  FPBits<T> bits(x);

  // If x is infinity NaN or zero, return it.
  if (bits.isInfOrNaN() || bits.isZero())
    return x;

  bool isNeg = bits.getSign();
  int exponent = bits.getExponent();
  int roundingMode = getRound();

  // If the exponent is greater than the most negative mantissa
  // exponent, then x is already an integer.
  if (exponent >= static_cast<int>(MantissaWidth<T>::value))
    return x;

  if (exponent <= -1) {
    switch (roundingMode) {
    case FE_DOWNWARD:
      return isNeg ? T(-1.0) : T(0.0);
    case FE_UPWARD:
      return isNeg ? T(-0.0) : T(1.0);
    case FE_TOWARDZERO:
      return isNeg ? T(-0.0) : T(0.0);
    case FE_TONEAREST:
      if (exponent <= -2 || bits.getMantissa() == 0)
        return isNeg ? T(-0.0) : T(0.0); // abs(x) <= 0.5
      else
        return isNeg ? T(-1.0) : T(1.0); // abs(x) > 0.5
    default:
      __builtin_unreachable();
    }
  }

  uint32_t trimSize = MantissaWidth<T>::value - exponent;
  FPBits<T> newBits = bits;
  newBits.setMantissa((bits.getMantissa() >> trimSize) << trimSize);
  T truncValue = T(newBits);

  // If x is already an integer, return it.
  if (truncValue == x)
    return x;

  UIntType trimValue = bits.getMantissa() & ((UIntType(1) << trimSize) - 1);
  UIntType halfValue = (UIntType(1) << (trimSize - 1));
  // If exponent is 0, trimSize will be equal to the mantissa width, and
  // truncIsOdd` will not be correct. So, we handle it as a special case
  // below.
  UIntType truncIsOdd = newBits.getMantissa() & (UIntType(1) << trimSize);

  switch (roundingMode) {
  case FE_DOWNWARD:
    return isNeg ? truncValue - T(1.0) : truncValue;
  case FE_UPWARD:
    return isNeg ? truncValue : truncValue + T(1.0);
  case FE_TOWARDZERO:
    return truncValue;
  case FE_TONEAREST:
    if (trimValue > halfValue) {
      return isNeg ? truncValue - T(1.0) : truncValue + T(1.0);
    } else if (trimValue == halfValue) {
      if (exponent == 0)
        return isNeg ? T(-2.0) : T(2.0);
      if (truncIsOdd)
        return isNeg ? truncValue - T(1.0) : truncValue + T(1.0);
      else
        return truncValue;
    } else {
      return truncValue;
    }
  default:
    __builtin_unreachable();
  }
}

namespace internal {

template <typename F, typename I,
          cpp::EnableIfType<cpp::IsFloatingPointType<F>::Value &&
                                cpp::IsIntegral<I>::Value,
                            int> = 0>
static inline I roundedFloatToSignedInteger(F x) {
  constexpr I IntegerMin = (I(1) << (sizeof(I) * 8 - 1));
  constexpr I IntegerMax = -(IntegerMin + 1);
  FPBits<F> bits(x);
  auto setDomainErrorAndRaiseInvalid = []() {
#if math_errhandling & MATH_ERRNO
    errno = EDOM; // NOLINT
#endif
#if math_errhandling & MATH_ERREXCEPT
    raiseExcept(FE_INVALID);
#endif
  };

  if (bits.isInfOrNaN()) {
    setDomainErrorAndRaiseInvalid();
    return bits.getSign() ? IntegerMin : IntegerMax;
  }

  int exponent = bits.getExponent();
  constexpr int exponentLimit = sizeof(I) * 8 - 1;
  if (exponent > exponentLimit) {
    setDomainErrorAndRaiseInvalid();
    return bits.getSign() ? IntegerMin : IntegerMax;
  } else if (exponent == exponentLimit) {
    if (bits.getSign() == 0 || bits.getMantissa() != 0) {
      setDomainErrorAndRaiseInvalid();
      return bits.getSign() ? IntegerMin : IntegerMax;
    }
    // If the control reaches here, then it means that the rounded
    // value is the most negative number for the signed integer type I.
  }

  // For all other cases, if `x` can fit in the integer type `I`,
  // we just return `x`. Implicit conversion will convert the
  // floating point value to the exact integer value.
  return x;
}

} // namespace internal

template <typename F, typename I,
          cpp::EnableIfType<cpp::IsFloatingPointType<F>::Value &&
                                cpp::IsIntegral<I>::Value,
                            int> = 0>
static inline I roundToSignedInteger(F x) {
  return internal::roundedFloatToSignedInteger<F, I>(round(x));
}

template <typename F, typename I,
          cpp::EnableIfType<cpp::IsFloatingPointType<F>::Value &&
                                cpp::IsIntegral<I>::Value,
                            int> = 0>
static inline I roundToSignedIntegerUsingCurrentRoundingMode(F x) {
  return internal::roundedFloatToSignedInteger<F, I>(
      roundUsingCurrentRoundingMode(x));
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_NEAREST_INTEGER_OPERATIONS_H

//===-- Floating-point manipulation functions -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_MANIPULATION_FUNCTIONS_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_MANIPULATION_FUNCTIONS_H

#include "FPBits.h"
#include "NearestIntegerOperations.h"
#include "NormalFloat.h"
#include "PlatformDefs.h"

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

#include <limits.h>
#include <math.h>

namespace __llvm_libc {
namespace fputil {

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T frexp(T x, int &exp) {
  FPBits<T> bits(x);
  if (bits.is_inf_or_nan())
    return x;
  if (bits.is_zero()) {
    exp = 0;
    return x;
  }

  NormalFloat<T> normal(bits);
  exp = normal.exponent + 1;
  normal.exponent = -1;
  return normal;
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T modf(T x, T &iptr) {
  FPBits<T> bits(x);
  if (bits.is_zero() || bits.is_nan()) {
    iptr = x;
    return x;
  } else if (bits.is_inf()) {
    iptr = x;
    return bits.get_sign() ? T(FPBits<T>::neg_zero()) : T(FPBits<T>::zero());
  } else {
    iptr = trunc(x);
    if (x == iptr) {
      // If x is already an integer value, then return zero with the right
      // sign.
      return bits.get_sign() ? T(FPBits<T>::neg_zero()) : T(FPBits<T>::zero());
    } else {
      return x - iptr;
    }
  }
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T copysign(T x, T y) {
  FPBits<T> xbits(x);
  xbits.set_sign(FPBits<T>(y).get_sign());
  return T(xbits);
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE int ilogb(T x) {
  // TODO: Raise appropriate floating point exceptions and set errno to the
  // an appropriate error value wherever relevant.
  FPBits<T> bits(x);
  if (bits.is_zero()) {
    return FP_ILOGB0;
  } else if (bits.is_nan()) {
    return FP_ILOGBNAN;
  } else if (bits.is_inf()) {
    return INT_MAX;
  }

  NormalFloat<T> normal(bits);
  // The C standard does not specify the return value when an exponent is
  // out of int range. However, XSI conformance required that INT_MAX or
  // INT_MIN are returned.
  // NOTE: It is highly unlikely that exponent will be out of int range as
  // the exponent is only 15 bits wide even for the 128-bit floating point
  // format.
  if (normal.exponent > INT_MAX)
    return INT_MAX;
  else if (normal.exponent < INT_MIN)
    return INT_MIN;
  else
    return normal.exponent;
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T logb(T x) {
  FPBits<T> bits(x);
  if (bits.is_zero()) {
    // TODO(Floating point exception): Raise div-by-zero exception.
    // TODO(errno): POSIX requires setting errno to ERANGE.
    return T(FPBits<T>::neg_inf());
  } else if (bits.is_nan()) {
    return x;
  } else if (bits.is_inf()) {
    // Return positive infinity.
    return T(FPBits<T>::inf());
  }

  NormalFloat<T> normal(bits);
  return normal.exponent;
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T ldexp(T x, int exp) {
  if (LIBC_UNLIKELY(exp == 0))
    return x;
  FPBits<T> bits(x);
  if (LIBC_UNLIKELY(bits.is_zero() || bits.is_inf_or_nan()))
    return x;

  // NormalFloat uses int32_t to store the true exponent value. We should ensure
  // that adding |exp| to it does not lead to integer rollover. But, if |exp|
  // value is larger the exponent range for type T, then we can return infinity
  // early. Because the result of the ldexp operation can be a subnormal number,
  // we need to accommodate the (mantissaWidht + 1) worth of shift in
  // calculating the limit.
  int exp_limit = FPBits<T>::MAX_EXPONENT + MantissaWidth<T>::VALUE + 1;
  if (exp > exp_limit)
    return bits.get_sign() ? T(FPBits<T>::neg_inf()) : T(FPBits<T>::inf());

  // Similarly on the negative side we return zero early if |exp| is too small.
  if (exp < -exp_limit)
    return bits.get_sign() ? T(FPBits<T>::neg_zero()) : T(FPBits<T>::zero());

  // For all other values, NormalFloat to T conversion handles it the right way.
  NormalFloat<T> normal(bits);
  normal.exponent += exp;
  return normal;
}

template <typename T, cpp::enable_if_t<cpp::is_floating_point_v<T>, int> = 0>
LIBC_INLINE T nextafter(T from, T to) {
  FPBits<T> from_bits(from);
  if (from_bits.is_nan())
    return from;

  FPBits<T> to_bits(to);
  if (to_bits.is_nan())
    return to;

  if (from == to)
    return to;

  using UIntType = typename FPBits<T>::UIntType;
  UIntType int_val = from_bits.uintval();
  UIntType sign_mask = (UIntType(1) << (sizeof(T) * 8 - 1));
  if (from != T(0.0)) {
    if ((from < to) == (from > T(0.0))) {
      ++int_val;
    } else {
      --int_val;
    }
  } else {
    int_val = (to_bits.uintval() & sign_mask) + UIntType(1);
  }

  return cpp::bit_cast<T>(int_val);
  // TODO: Raise floating point exceptions as required by the standard.
}

} // namespace fputil
} // namespace __llvm_libc

#ifdef SPECIAL_X86_LONG_DOUBLE
#include "x86_64/NextAfterLongDouble.h"
#endif // SPECIAL_X86_LONG_DOUBLE

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_MANIPULATION_FUNCTIONS_H

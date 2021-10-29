//===-- Floating point divsion and remainder operations ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_DIVISION_AND_REMAINDER_OPERATIONS_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_DIVISION_AND_REMAINDER_OPERATIONS_H

#include "FPBits.h"
#include "ManipulationFunctions.h"
#include "NormalFloat.h"

#include "src/__support/CPP/TypeTraits.h"

namespace __llvm_libc {
namespace fputil {

static constexpr int quotientLSBBits = 3;

// The implementation is a bit-by-bit algorithm which uses integer division
// to evaluate the quotient and remainder.
template <typename T,
          cpp::EnableIfType<cpp::IsFloatingPointType<T>::Value, int> = 0>
static inline T remquo(T x, T y, int &q) {
  FPBits<T> xbits(x), ybits(y);
  if (xbits.isNaN())
    return x;
  if (ybits.isNaN())
    return y;
  if (xbits.isInf() || ybits.isZero())
    return FPBits<T>::buildNaN(1);

  if (xbits.isZero()) {
    q = 0;
    return __llvm_libc::fputil::copysign(T(0.0), x);
  }

  if (ybits.isInf()) {
    q = 0;
    return x;
  }

  bool resultSign = (xbits.getSign() == ybits.getSign() ? false : true);

  // Once we know the sign of the result, we can just operate on the absolute
  // values. The correct sign can be applied to the result after the result
  // is evaluated.
  xbits.setSign(0);
  ybits.setSign(0);

  NormalFloat<T> normalx(xbits), normaly(ybits);
  int exp = normalx.exponent - normaly.exponent;
  typename NormalFloat<T>::UIntType mx = normalx.mantissa,
                                    my = normaly.mantissa;

  q = 0;
  while (exp >= 0) {
    unsigned shiftCount = 0;
    typename NormalFloat<T>::UIntType n = mx;
    for (shiftCount = 0; n < my; n <<= 1, ++shiftCount)
      ;

    if (static_cast<int>(shiftCount) > exp)
      break;

    exp -= shiftCount;
    if (0 <= exp && exp < quotientLSBBits)
      q |= (1 << exp);

    mx = n - my;
    if (mx == 0) {
      q = resultSign ? -q : q;
      return __llvm_libc::fputil::copysign(T(0.0), x);
    }
  }

  NormalFloat<T> remainder(exp + normaly.exponent, mx, 0);

  // Since NormalFloat to native type conversion is a truncation operation
  // currently, the remainder value in the native type is correct as is.
  // However, if NormalFloat to native type conversion is updated in future,
  // then the conversion to native remainder value should be updated
  // appropriately and some directed tests added.
  T nativeRemainder(remainder);
  T absy = T(ybits);
  int cmp = remainder.mul2(1).cmp(normaly);
  if (cmp > 0) {
    q = q + 1;
    if (x >= T(0.0))
      nativeRemainder = nativeRemainder - absy;
    else
      nativeRemainder = absy - nativeRemainder;
  } else if (cmp == 0) {
    if (q & 1) {
      q += 1;
      if (x >= T(0.0))
        nativeRemainder = -nativeRemainder;
    } else {
      if (x < T(0.0))
        nativeRemainder = -nativeRemainder;
    }
  } else {
    if (x < T(0.0))
      nativeRemainder = -nativeRemainder;
  }

  q = resultSign ? -q : q;
  if (nativeRemainder == T(0.0))
    return __llvm_libc::fputil::copysign(T(0.0), x);
  return nativeRemainder;
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_DIVISION_AND_REMAINDER_OPERATIONS_H

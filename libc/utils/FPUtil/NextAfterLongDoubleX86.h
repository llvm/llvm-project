//===-- nextafter implementation for x86 long double numbers ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_NEXT_AFTER_LONG_DOUBLE_X86_H
#define LLVM_LIBC_UTILS_FPUTIL_NEXT_AFTER_LONG_DOUBLE_X86_H

#include "FPBits.h"

#include <stdint.h>

namespace __llvm_libc {
namespace fputil {

static inline long double nextafter(long double from, long double to) {
  using FPBits = FPBits<long double>;
  FPBits fromBits(from);
  if (fromBits.isNaN())
    return from;

  FPBits toBits(to);
  if (toBits.isNaN())
    return to;

  if (from == to)
    return to;

  // Convert pseudo subnormal number to normal number.
  if (fromBits.implicitBit == 1 && fromBits.exponent == 0) {
    fromBits.exponent = 1;
  }

  using UIntType = FPBits::UIntType;
  constexpr UIntType signVal = (UIntType(1) << 79);
  constexpr UIntType mantissaMask =
      (UIntType(1) << MantissaWidth<long double>::value) - 1;
  auto intVal = fromBits.bitsAsUInt();
  if (from < 0.0l) {
    if (from > to) {
      if (intVal == (signVal + FPBits::maxSubnormal)) {
        // We deal with normal/subnormal boundary separately to avoid
        // dealing with the implicit bit.
        intVal = signVal + FPBits::minNormal;
      } else if ((intVal & mantissaMask) == mantissaMask) {
        fromBits.mantissa = 0;
        // Incrementing exponent might overflow the value to infinity,
        // which is what is expected. Since NaNs are handling separately,
        // it will never overflow "beyond" infinity.
        ++fromBits.exponent;
        return fromBits;
      } else {
        ++intVal;
      }
    } else {
      if (intVal == (signVal + FPBits::minNormal)) {
        // We deal with normal/subnormal boundary separately to avoid
        // dealing with the implicit bit.
        intVal = signVal + FPBits::maxSubnormal;
      } else if ((intVal & mantissaMask) == 0) {
        fromBits.mantissa = mantissaMask;
        // from == 0 is handled separately so decrementing the exponent will not
        // lead to underflow.
        --fromBits.exponent;
        return fromBits;
      } else {
        --intVal;
      }
    }
  } else if (from == 0.0l) {
    if (from > to)
      intVal = signVal + 1;
    else
      intVal = 1;
  } else {
    if (from > to) {
      if (intVal == FPBits::minNormal) {
        intVal = FPBits::maxSubnormal;
      } else if ((intVal & mantissaMask) == 0) {
        fromBits.mantissa = mantissaMask;
        // from == 0 is handled separately so decrementing the exponent will not
        // lead to underflow.
        --fromBits.exponent;
        return fromBits;
      } else {
        --intVal;
      }
    } else {
      if (intVal == FPBits::maxSubnormal) {
        intVal = FPBits::minNormal;
      } else if ((intVal & mantissaMask) == mantissaMask) {
        fromBits.mantissa = 0;
        // Incrementing exponent might overflow the value to infinity,
        // which is what is expected. Since NaNs are handling separately,
        // it will never overflow "beyond" infinity.
        ++fromBits.exponent;
        return fromBits;
      } else {
        ++intVal;
      }
    }
  }

  return *reinterpret_cast<long double *>(&intVal);
  // TODO: Raise floating point exceptions as required by the standard.
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_NEXT_AFTER_LONG_DOUBLE_X86_H

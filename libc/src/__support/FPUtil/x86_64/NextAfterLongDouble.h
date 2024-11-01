//===-- nextafter implementation for x86 long double numbers ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_NEXT_AFTER_LONG_DOUBLE_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_NEXT_AFTER_LONG_DOUBLE_H

#include "src/__support/macros/properties/architectures.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86)
#error "Invalid include"
#endif

#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/FPBits.h"

#include <stdint.h>

namespace __llvm_libc {
namespace fputil {

LIBC_INLINE long double nextafter(long double from, long double to) {
  using FPBits = FPBits<long double>;
  FPBits from_bits(from);
  if (from_bits.is_nan())
    return from;

  FPBits to_bits(to);
  if (to_bits.is_nan())
    return to;

  if (from == to)
    return to;

  // Convert pseudo subnormal number to normal number.
  if (from_bits.get_implicit_bit() == 1 &&
      from_bits.get_unbiased_exponent() == 0) {
    from_bits.set_unbiased_exponent(1);
  }

  using UIntType = FPBits::UIntType;
  constexpr UIntType SIGN_VAL = (UIntType(1) << 79);
  constexpr UIntType MANTISSA_MASK =
      (UIntType(1) << MantissaWidth<long double>::VALUE) - 1;
  UIntType int_val = from_bits.uintval();
  if (from < 0.0l) {
    if (from > to) {
      if (int_val == (SIGN_VAL + FPBits::MAX_SUBNORMAL)) {
        // We deal with normal/subnormal boundary separately to avoid
        // dealing with the implicit bit.
        int_val = SIGN_VAL + FPBits::MIN_NORMAL;
      } else if ((int_val & MANTISSA_MASK) == MANTISSA_MASK) {
        from_bits.set_mantissa(0);
        // Incrementing exponent might overflow the value to infinity,
        // which is what is expected. Since NaNs are handling separately,
        // it will never overflow "beyond" infinity.
        from_bits.set_unbiased_exponent(from_bits.get_unbiased_exponent() + 1);
        return from_bits;
      } else {
        ++int_val;
      }
    } else {
      if (int_val == (SIGN_VAL + FPBits::MIN_NORMAL)) {
        // We deal with normal/subnormal boundary separately to avoid
        // dealing with the implicit bit.
        int_val = SIGN_VAL + FPBits::MAX_SUBNORMAL;
      } else if ((int_val & MANTISSA_MASK) == 0) {
        from_bits.set_mantissa(MANTISSA_MASK);
        // from == 0 is handled separately so decrementing the exponent will not
        // lead to underflow.
        from_bits.set_unbiased_exponent(from_bits.get_unbiased_exponent() - 1);
        return from_bits;
      } else {
        --int_val;
      }
    }
  } else if (from == 0.0l) {
    if (from > to)
      int_val = SIGN_VAL + 1;
    else
      int_val = 1;
  } else {
    if (from > to) {
      if (int_val == FPBits::MIN_NORMAL) {
        int_val = FPBits::MAX_SUBNORMAL;
      } else if ((int_val & MANTISSA_MASK) == 0) {
        from_bits.set_mantissa(MANTISSA_MASK);
        // from == 0 is handled separately so decrementing the exponent will not
        // lead to underflow.
        from_bits.set_unbiased_exponent(from_bits.get_unbiased_exponent() - 1);
        return from_bits;
      } else {
        --int_val;
      }
    } else {
      if (int_val == FPBits::MAX_SUBNORMAL) {
        int_val = FPBits::MIN_NORMAL;
      } else if ((int_val & MANTISSA_MASK) == MANTISSA_MASK) {
        from_bits.set_mantissa(0);
        // Incrementing exponent might overflow the value to infinity,
        // which is what is expected. Since NaNs are handling separately,
        // it will never overflow "beyond" infinity.
        from_bits.set_unbiased_exponent(from_bits.get_unbiased_exponent() + 1);
        return from_bits;
      } else {
        ++int_val;
      }
    }
  }

  return cpp::bit_cast<long double>(int_val);
  // TODO: Raise floating point exceptions as required by the standard.
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_X86_64_NEXT_AFTER_LONG_DOUBLE_H

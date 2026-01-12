//===-- Implementation header for llogb --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_LLOGB_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_LLOGB_H

#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"  // LIBC_UNLIKELY
#include <limits.h>

namespace LIBC_NAMESPACE_DECL {
namespace math {

LIBC_INLINE constexpr long llogb(double x) {
  using FPBits = fputil::FPBits<double>;
  FPBits bits(x);

  if (LIBC_UNLIKELY(bits.is_zero())) {
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return -LONG_MAX - 1; // FP_LLOGB0
  }

  if (LIBC_UNLIKELY(bits.is_inf())) {
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return LONG_MAX;
  }

  if (LIBC_UNLIKELY(bits.is_nan())) {
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
#ifdef __FP_LOGBNAN_MIN
    return -LONG_MAX - 1;
#else
    return LONG_MAX; // FP_LLOGBNAN
#endif
  }

  int biased_exp = bits.get_biased_exponent();

  // Handle normal numbers
  // Note: For double, the exponent range (-1022 to 1023) always fits in long,
  // so no range check is needed here.
  if (biased_exp > 0) {
    return biased_exp - FPBits::EXP_BIAS;
  }

  // Handle subnormal numbers
  // Formula: log2(mantissa * 2^-1074) = log2(mantissa) - 1074
  uint64_t mantissa = bits.get_mantissa();

  // Define constants
  constexpr int EXP_ADJUSTMENT = FPBits::EXP_BIAS + FPBits::FRACTION_LEN - 1;
  constexpr int STORAGE_MSB = (sizeof(uint64_t) * CHAR_BIT) - 1;

  // log2(mantissa) = STORAGE_MSB - countl_zero(mantissa)
  return (STORAGE_MSB - cpp::countl_zero(mantissa)) - EXP_ADJUSTMENT;
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_LLOGB_H

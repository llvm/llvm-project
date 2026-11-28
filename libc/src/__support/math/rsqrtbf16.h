//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of rsqrtbf16.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_RSQRTBF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_RSQRTBF16_H

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {

LIBC_INLINE bfloat16 rsqrtbf16(bfloat16 x) {
  using FPBits = fputil::FPBits<bfloat16>;
  FPBits bits(x);

  if (LIBC_UNLIKELY(bits.is_zero() || bits.is_inf_or_nan() || bits.is_neg())) {
    // x is NaN
    if (bits.is_nan()) {
      if (bits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      return x;
    }

    // |x| = 0
    if (bits.is_zero()) {
      fputil::raise_except_if_required(FE_DIVBYZERO);
      fputil::set_errno_if_required(ERANGE);
      return FPBits::inf(bits.sign()).get_val();
    }

    // -inf <= x < 0
    if (bits.is_neg()) {
      fputil::raise_except_if_required(FE_INVALID);
      fputil::set_errno_if_required(EDOM);
      return FPBits::quiet_nan().get_val();
    }

    // x = +inf
    return FPBits::zero().get_val();
  }

  float xf = fputil::cast<float>(x);
  return fputil::cast<bfloat16>(1.0f / fputil::sqrt<float>(xf));
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_RSQRTBF16_H

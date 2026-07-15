//===-- e^x range reduction and evaluation using integer-only --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: update the description of this file above

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_UTILS_H

#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/big_int.h"
#include "src/__support/frac64.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math_extras.h"

#undef LIBC_TARGET_IS_BIG_ENDIAN
#if !defined(__BYTE_ORDER__) || !defined(__ORDER_LITTLE_ENDIAN__) ||           \
    !defined(__ORDER_BIG_ENDIAN__)
#define LIBC_TARGET_IS_BIG_ENDIAN 0
#else
#define LIBC_TARGET_IS_BIG_ENDIAN (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#endif // LIBC_TARGET_IS_BIG_ENDIAN

namespace LIBC_NAMESPACE_DECL {

namespace math {

namespace integer_only {

// round(1/log(2), D, RN)
LIBC_INLINE_VAR constexpr double INV_LN2 = 0x1.71547652b82fep+0;

// TODO: Remez eval
// TODO: test around and see what at what degree the accuracy is "acceptable"
LIBC_INLINE_VAR constexpr Frac64 EXPF_COEFFS[6] = {};

LIBC_INLINE constexpr float
expf_range_reduction([[maybe_unused]] const float &x) {
  // TODO: declare the range reduction params and this function
  return 0.0f;
}

LIBC_INLINE constexpr float expf_eval() {
  // TODO: declare params and this function
  return 0.0f;
}

} // namespace integer_only

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_UTILS_H

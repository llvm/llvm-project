//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains utilities for implementing integer-only implementation
/// of expf(x)
///
//===----------------------------------------------------------------------===//

// TODO: this file is expected to be used for exp*(x) functions, not only
// limited to expf(x)

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_UTILS_H

#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/big_int.h"
#include "src/__support/frac64.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math_extras.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

namespace integer_only {

// print(2+round(1/log(2), 64, RN));
// LSB(INV_LN2) = 2^-63
LIBC_INLINE_VAR constexpr Frac64 INV_LN2 = Frac64(0xb8aa'3b29'5c17'f0bc);

// 64-bit polynomial approximation of 2^x coefficients generated with Sollya:
// > P = fpminimax(2^x, 11, [|1, 64...|], [0, 1], absolute, fixed);
// Store the fractional part of the coefficients below
// > dirtyinfnorm(2^x - P(x), [0, 1]);
// 0x1.6238...p-58
// LSB(EXPF_COEFFS[i]) = 2^-64
LIBC_INLINE_VAR constexpr Frac64 EXPF_COEFFS[] = {
    Frac64(0xb172'17f7'd1cf'b7cf), // x
    Frac64(0x3d7f'7bff'057d'4a5e), // x^2
    Frac64(0x0e35'846b'8363'9484), // x^3
    Frac64(0x0276'556d'ec97'dcd4), // x^4
    Frac64(0x0057'61ff'dc04'c7ff), // x^5
    Frac64(0x000a'1847'b6e7'92ec), // x^6
    Frac64(0x0000'ffe8'14e5'7033), // x^7
    Frac64(0x0000'1628'b6e9'70c8), // x^8
    Frac64(0x0000'01b8'8ce7'4088), // x^9
    Frac64(0x0000'001c'18d5'cb29), // x^10
    Frac64(0x0000'0002'b43f'4490), // x^11
};

} // namespace integer_only

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_UTILS_H

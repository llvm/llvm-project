//===-- e^x integer-only utility functions ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_UTILS_H

#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/big_int.h"
#include "src/__support/frac32.h"
#include "src/__support/frac64.h"
#include "src/__support/macros/attributes.h"
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

// print(2+round(1/log(2), 64, RN));
// LSB(INV_LN2) = 2^-63
LIBC_INLINE_VAR constexpr Frac64 INV_LN2 = Frac64(0xb8aa'3b29'5c17'f0bc);

// 1-ULP
// 64-bit polynomial approximation of 2^x coefficients generated with Sollya:
// > P = fpminimax(2^x, 6, [|1, 64...|], [0, 1], absolute, fixed);
// Store the fractional part of the coefficients below
// > dirtyinfnorm(2^x - P(x), [0, 1]);
// 0x1.9d39...p-29
// ULPs of coeffs = 2^-64
LIBC_INLINE_VAR constexpr Frac64 EXPF_COEFFS[] = {
    Frac64(0xb172'14ea'215c'7750), // x
    Frac64(0x3d7f'b5e7'4e78'9f2b), // x^2
    Frac64(0x0e34'15ac'7481'5dee), // x^3
    Frac64(0x027a'7e40'a2eb'6584), // x^4
    Frac64(0x0051'56c0'9d53'15f3), // x^5
    Frac64(0x000e'4a74'a170'46e8), // x^6
};

// print(2+round(1/log(2), 32, RN));
// LSB(INV_LN2_FRAC32) = 2^-31
LIBC_INLINE_VAR constexpr Frac32 INV_LN2_FRAC32 = Frac32(0xb8aa'3b29);

// 1-ULP
// Degree-6 still works fine (p-29), degree-5 yields p-23 accuracy
// This is still the case for Frac32!
// 32-bit polynomial approximation of 2^x coefficients generated with Sollya:
// > P = fpminimax(2^x, 6, [|1, 32...|], [0, 1], absolute, fixed);
// Store the fractional part of the coefficients below
// > dirtyinfnorm(2^x - P(x), [0, 1]);
// 0x1.9ded...p-29
// ULPs of coeffs = 2^-32
LIBC_INLINE_VAR constexpr Frac32 EXPF_COEFFS_FRAC32[] = {
    Frac32(0xb172'14e8), Frac32(0x3d7f'b5f8), Frac32(0x0e34'1554),
    Frac32(0x027a'7f04), Frac32(0x0051'55fe), Frac32(0x000e'4abc)};

} // namespace integer_only

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_UTILS_H

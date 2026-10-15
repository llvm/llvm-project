//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the look-up tables for integer-only, statically rounded
/// exp*(x) functions
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_CONSTANTS_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_CONSTANTS_H

#include "src/__support/frac128.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

namespace static_rounding {

// print(2+round(1/log(2), 128, RN));
// LSB(INV_LN2) = 2^-127
LIBC_INLINE_VAR constexpr Frac128 INV_LN2_F128 =
    Frac128({0xbe87'fed0'691d'3e89ULL, 0xb8aa'3b29'5c17'f0bbULL});

// 2^x for x from 0 to (0b1111/2^4) = 15/16
// > for i from 0 to 15 do {
//   print(1+round(2^(i/16), 128, RN));
// };
// LSB(EXP_MID4[i]) = 2^-127
LIBC_INLINE_VAR constexpr Frac128 EXP_MID[] = {
    Frac128({0x0000'0000'0000'0000ULL, 0x8000'0000'0000'0000ULL}),
    Frac128({0xc5c9'5b8c'2154'c1b2ULL, 0x85aa'c367'cc48'7b14ULL}),
    Frac128({0xfbe4'6287'58a5'3c90ULL, 0x8b95'c1e3'ea8b'd6e6ULL}),
    Frac128({0x0fd6'd8e0'ae5a'c9d8ULL, 0x91c3'd373'ab11'c336ULL}),
    Frac128({0x46ad'2318'2e42'f6f6ULL, 0x9837'f051'8db8'a96fULL}),
    Frac128({0xa091'1f09'ebb9'fdd1ULL, 0x9ef5'3260'91a1'11adULL}),
    Frac128({0x1cbd'7f62'1710'701bULL, 0xa5fe'd6a9'b151'38eaULL}),
    Frac128({0x4980'a8c8'f59a'2ec4ULL, 0xad58'3eea'42a1'4ac6ULL}),
    Frac128({0x597d'89b3'754a'be9fULL, 0xb504'f333'f9de'6484ULL}),
    Frac128({0xa881'1fb6'6d0f'af7aULL, 0xbd08'a39f'580c'36beULL}),
    Frac128({0x3e2a'd0c9'64dd'9f37ULL, 0xc567'2a11'5506'daddULL}),
    Frac128({0xe235'838f'95f2'c6edULL, 0xce24'8c15'1f84'80e3ULL}),
    Frac128({0x39a6'8bb9'902d'3fdeULL, 0xd744'fcca'd69d'6af4ULL}),
    Frac128({0x0658'9504'8dd3'33caULL, 0xe0cc'deec'2a94'e111ULL}),
    Frac128({0xd02d'75b3'706e'54fbULL, 0xeac0'c6e7'dd24'392eULL}),
    Frac128({0x7b9d'0c7a'ed98'0fc3ULL, 0xf525'7d15'2486'cc2cULL}),
};

// 128-bit polynomial approximation of 2^x coefficients generated with Sollya:
// > P = fpminimax(2^x, 12, [|1, 128...|], [0, 1/16], absolute, fixed);
// Store the fractional part of the coefficients below
// > dirtyinfnorm(2^x - P(x), [0, 1/16]);
// 0x1.b328...p-117
// LSB(EXPF_COEFFS[i]) = 2^-128
LIBC_INLINE_VAR constexpr Frac128 EXP_COEFFS[] = {
    // degree-0 = 1, add back afterwards to reduce calc ops
    Frac128({0xc9e3'b398'033f'0902ULL, 0xb172'17f7'd1cf'79abULL}),
    Frac128({0xde2d'60e0'866c'6365ULL, 0x3d7f'7bff'058b'1d50ULL}),
    Frac128({0x99d3'ad04'd47d'0efeULL, 0x0e35'846b'8250'5fc5ULL}),
    Frac128({0x399a'b423'1644'1b55ULL, 0x0276'556d'f749'cee5ULL}),
    Frac128({0x405f'ebef'0f76'011aULL, 0x0057'61ff'9e29'9cc4ULL}),
    Frac128({0x1ac4'9999'c0ef'b9abULL, 0x000a'1848'97c3'63c4ULL}),
    Frac128({0xe6f8'52f9'914f'2d9aULL, 0x0000'ffe5'fe2c'4573ULL}),
    Frac128({0x6056'28a6'5061'b43fULL, 0x0000'162c'0223'a816ULL}),
    Frac128({0x6b99'3efe'2193'e54cULL, 0x0000'01b5'253d'0671ULL}),
    Frac128({0x9c76'f49c'ed65'743aULL, 0x0000'001e'4cf8'0b70ULL}),
    Frac128({0x1ac1'3330'6c4e'3d33ULL, 0x0000'0001'e8ae'6938ULL}),
    Frac128({0x219f'9904'1f29'5f13ULL, 0x0000'0000'1cd9'af73ULL}),
};

// 64-bit polynomial approximation of 2^x coefficients generated with Sollya:
// > P = fpminimax(2^x, 6, [|1, 64...|], [0, 1/16], absolute, fixed);
// Store the fractional part of the coefficients below
// > dirtyinfnorm(2^x - P(x), [0, 1/16]);
// 0x1.2ac9...p-57
// This is different from EXPF_COEFFS: EXPF_COEFFS is for approximating for x in
// range of [0, 1]
// LSB(EXP_COEFFS[i]) = 2^-64
LIBC_INLINE_VAR constexpr Frac64 EXP_64_COEFFS[] = {
    // degree-0 = 1, add back afterwards to reduce calc ops
    Frac64(0xb172'17f7'd1cd'3e3cULL), Frac64(0x3d7f'7bff'0838'4d4aULL),
    Frac64(0x0e35'846a'6f46'd462ULL), Frac64(0x0276'55a0'd536'1dd9ULL),
    Frac64(0x0057'5d3d'4b45'0056ULL), Frac64(0x000a'504b'13fe'e008ULL),
};

// print(2+round(1/log(2), 64, RN));
// LSB(INV_LN2) = 2^-63
LIBC_INLINE_VAR constexpr Frac64 INV_LN2_F64 = Frac64(0xb8aa'3b29'5c17'f0bc);

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

} // namespace static_rounding

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_CONSTANTS_H

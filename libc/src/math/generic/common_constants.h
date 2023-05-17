//===-- Common constants for math functions ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_GENERIC_COMMON_CONSTANTS_H
#define LLVM_LIBC_SRC_MATH_GENERIC_COMMON_CONSTANTS_H

namespace __llvm_libc {

// Lookup table for (1/f) where f = 1 + n*2^(-7), n = 0..127.
extern const double ONE_OVER_F[128];

// Lookup table for log(f) = log(1 + n*2^(-7)) where n = 0..127.
extern const double LOG_F[128];

// Lookup table for range reduction constants r for logarithms.
extern const float R[128];

// Lookup table for range reduction constants r for logarithms.
extern const double RD[128];

// Lookup table for -log(r)
extern const double LOG_R[128];

// Lookup table for exp(m) with m = -104, ..., 89.
//   -104 = floor(log(single precision's min denormal))
//     89 = ceil(log(single precision's max normal))
// Table is generated with Sollya as follow:
// > display = hexadecimal;
// > for i from -104 to 89 do { D(exp(i)); };
extern const double EXP_M1[195];

// Lookup table for exp(m * 2^(-7)) with m = 0, ..., 127.
// Table is generated with Sollya as follow:
// > display = hexadecimal;
// > for i from 0 to 127 do { D(exp(i / 128)); };
extern const double EXP_M2[128];

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_GENERIC_COMMON_CONSTANTS_H

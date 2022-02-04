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

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_GENERIC_COMMON_CONSTANTS_H

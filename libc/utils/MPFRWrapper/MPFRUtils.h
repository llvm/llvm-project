//===-- MPFRUtils.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_TESTUTILS_MPFRUTILS_H
#define LLVM_LIBC_UTILS_TESTUTILS_MPFRUTILS_H

#include <stdint.h>

namespace __llvm_libc {
namespace testing {
namespace mpfr {

struct Tolerance {
  // Number of bits used to represent the fractional
  // part of a value of type 'float'.
  static constexpr unsigned int floatPrecision = 23;

  // Number of bits used to represent the fractional
  // part of a value of type 'double'.
  static constexpr unsigned int doublePrecision = 52;

  // The base precision of the number. For example, for values of
  // type float, the base precision is the value |floatPrecision|.
  unsigned int basePrecision;

  unsigned int width; // Number of valid LSB bits in |value|.

  // The bits in the tolerance value. The tolerance value will be
  // sum(bits[width - i] * 2 ^ (- basePrecision - i)) for |i| in
  // range [1, width].
  uint32_t bits;
};

// Return true if |libcOutput| is within the tolerance |t| of the cos(x)
// value as evaluated by MPFR.
bool equalsCos(float x, float libcOutput, const Tolerance &t);

// Return true if |libcOutput| is within the tolerance |t| of the sin(x)
// value as evaluated by MPFR.
bool equalsSin(float x, float libcOutput, const Tolerance &t);

} // namespace mpfr
} // namespace testing
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_TESTUTILS_MPFRUTILS_H

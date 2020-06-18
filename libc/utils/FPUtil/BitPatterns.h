//===-- Bit patterns of common floating point numbers -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_BIT_PATTERNS_H
#define LLVM_LIBC_UTILS_FPUTIL_BIT_PATTERNS_H

#include "FloatProperties.h"

#include <float.h>

static_assert(
    FLT_RADIX == 2,
    "LLVM libc only supports radix 2 IEEE 754 floating point formats.");

namespace __llvm_libc {
namespace fputil {

template <typename T> struct BitPatterns {};

template <> struct BitPatterns<float> {
  using BitsType = FloatProperties<float>::BitsType;

  static constexpr BitsType inf = 0x7f800000U;
  static constexpr BitsType negInf = 0xff800000U;

  static constexpr BitsType zero = 0x0;
  static constexpr BitsType negZero = 0x80000000U;

  static constexpr BitsType one = 0x3f800000U;

  // Examples of quiet NAN.
  static constexpr BitsType aQuietNaN = 0x7fc00000U;
  static constexpr BitsType aNegativeQuietNaN = 0xffc00000U;

  // Examples of signalling NAN.
  static constexpr BitsType aSignallingNaN = 0x7f800001U;
  static constexpr BitsType aNegativeSignallingNaN = 0xff800001U;
};

template <> struct BitPatterns<double> {
  using BitsType = FloatProperties<double>::BitsType;

  static constexpr BitsType inf = 0x7ff0000000000000ULL;
  static constexpr BitsType negInf = 0xfff0000000000000ULL;

  static constexpr BitsType zero = 0x0ULL;
  static constexpr BitsType negZero = 0x8000000000000000ULL;

  static constexpr BitsType one = 0x3FF0000000000000ULL;

  // Examples of quiet NAN.
  static constexpr BitsType aQuietNaN = 0x7ff8000000000000ULL;
  static constexpr BitsType aNegativeQuietNaN = 0xfff8000000000000ULL;

  // Examples of signalling NAN.
  static constexpr BitsType aSignallingNaN = 0x7ff0000000000001ULL;
  static constexpr BitsType aNegativeSignallingNaN = 0xfff0000000000001ULL;
};

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_BIT_PATTERNS_H

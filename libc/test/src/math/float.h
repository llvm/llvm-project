//===-- Single precision floating point test utils --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_FLOAT_H
#define LLVM_LIBC_TEST_SRC_MATH_FLOAT_H

#include "src/math/math_utils.h"

namespace __llvm_libc {
namespace testing {

struct FloatBits {
  // The various NaN bit patterns here are just one of the many possible
  // patterns. The functions isQNan and isNegQNan can help understand why.

  static const uint32_t QNan = 0x7fc00000;
  static const uint32_t NegQNan = 0xffc00000;

  static const uint32_t SNan = 0x7f800001;
  static const uint32_t NegSNan = 0xff800001;

  static bool isQNan(float f) {
    uint32_t bits = as_uint32_bits(f);
    return ((0x7fc00000 & bits) != 0) && ((0x80000000 & bits) == 0);
  }

  static bool isNegQNan(float f) {
    uint32_t bits = as_uint32_bits(f);
    return 0xffc00000 & bits;
  }

  static constexpr uint32_t Zero = 0x0;
  static constexpr uint32_t NegZero = 0x80000000;

  static constexpr uint32_t Inf = 0x7f800000;
  static constexpr uint32_t NegInf = 0xff800000;

  static constexpr uint32_t One = 0x3f800000;
};

} // namespace testing
} // namespace __llvm_libc

#endif // LLVM_LIBC_TEST_SRC_MATH_FLOAT_H

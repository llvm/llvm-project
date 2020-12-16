//===-- Unittests for x86 long double -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils/FPUtil/FPBits.h"
#include "utils/UnitTest/Test.h"

#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<long double>;

TEST(X86LongDoubleTest, isNaN) {
  // In the nan checks below, we use the macro isnan from math.h to ensure that
  // a number is actually a NaN. The isnan macro resolves to the compiler
  // builtin function. Hence, matching LLVM-libc's notion of NaN with the
  // isnan result ensures that LLVM-libc's behavior matches the compiler's
  // behavior.

  FPBits bits(0.0l);
  bits.exponent = FPBits::maxExponent;
  for (unsigned int i = 0; i < 1000000; ++i) {
    // If exponent has the max value and the implicit bit is 0,
    // then the number is a NaN for all values of mantissa.
    bits.mantissa = i;
    long double nan = bits;
    ASSERT_NE(isnan(nan), 0);
    ASSERT_TRUE(bits.isNaN());
  }

  bits.implicitBit = 1;
  for (unsigned int i = 1; i < 1000000; ++i) {
    // If exponent has the max value and the implicit bit is 1,
    // then the number is a NaN for all non-zero values of mantissa.
    // Note the initial value of |i| of 1 to avoid a zero mantissa.
    bits.mantissa = i;
    long double nan = bits;
    ASSERT_NE(isnan(nan), 0);
    ASSERT_TRUE(bits.isNaN());
  }

  bits.exponent = 1;
  bits.implicitBit = 0;
  for (unsigned int i = 0; i < 1000000; ++i) {
    // If exponent is non-zero and also not max, and the implicit bit is 0,
    // then the number is a NaN for all values of mantissa.
    bits.mantissa = i;
    long double nan = bits;
    ASSERT_NE(isnan(nan), 0);
    ASSERT_TRUE(bits.isNaN());
  }

  bits.exponent = 1;
  bits.implicitBit = 1;
  for (unsigned int i = 0; i < 1000000; ++i) {
    // If exponent is non-zero and also not max, and the implicit bit is 1,
    // then the number is normal value for all values of mantissa.
    bits.mantissa = i;
    long double valid = bits;
    ASSERT_EQ(isnan(valid), 0);
    ASSERT_FALSE(bits.isNaN());
  }

  bits.exponent = 0;
  bits.implicitBit = 1;
  for (unsigned int i = 0; i < 1000000; ++i) {
    // If exponent is zero, then the number is a valid but denormal value.
    bits.mantissa = i;
    long double valid = bits;
    ASSERT_EQ(isnan(valid), 0);
    ASSERT_FALSE(bits.isNaN());
  }

  bits.exponent = 0;
  bits.implicitBit = 0;
  for (unsigned int i = 0; i < 1000000; ++i) {
    // If exponent is zero, then the number is a valid but denormal value.
    bits.mantissa = i;
    long double valid = bits;
    ASSERT_EQ(isnan(valid), 0);
    ASSERT_FALSE(bits.isNaN());
  }
}

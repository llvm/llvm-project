//===-- Unittests for x86 long double -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/Test.h"

#include <math.h>

using FPBits = LIBC_NAMESPACE::fputil::FPBits<long double>;

TEST(LlvmLibcX86LongDoubleTest, is_nan) {
  // In the nan checks below, we use the macro isnan from math.h to ensure that
  // a number is actually a NaN. The isnan macro resolves to the compiler
  // builtin function. Hence, matching LLVM-libc's notion of NaN with the
  // isnan result ensures that LLVM-libc's behavior matches the compiler's
  // behavior.
  constexpr uint32_t COUNT = 100'000;

  FPBits bits(0.0l);
  bits.set_biased_exponent(FPBits::MAX_BIASED_EXPONENT);
  for (unsigned int i = 0; i < COUNT; ++i) {
    // If exponent has the max value and the implicit bit is 0,
    // then the number is a NaN for all values of mantissa.
    bits.set_mantissa(i);
    long double nan = bits;
    ASSERT_NE(static_cast<int>(isnan(nan)), 0);
    ASSERT_TRUE(bits.is_nan());
  }

  bits.set_implicit_bit(1);
  for (unsigned int i = 1; i < COUNT; ++i) {
    // If exponent has the max value and the implicit bit is 1,
    // then the number is a NaN for all non-zero values of mantissa.
    // Note the initial value of |i| of 1 to avoid a zero mantissa.
    bits.set_mantissa(i);
    long double nan = bits;
    ASSERT_NE(static_cast<int>(isnan(nan)), 0);
    ASSERT_TRUE(bits.is_nan());
  }

  bits.set_biased_exponent(1);
  bits.set_implicit_bit(0);
  for (unsigned int i = 0; i < COUNT; ++i) {
    // If exponent is non-zero and also not max, and the implicit bit is 0,
    // then the number is a NaN for all values of mantissa.
    bits.set_mantissa(i);
    long double nan = bits;
    ASSERT_NE(static_cast<int>(isnan(nan)), 0);
    ASSERT_TRUE(bits.is_nan());
  }

  bits.set_biased_exponent(1);
  bits.set_implicit_bit(1);
  for (unsigned int i = 0; i < COUNT; ++i) {
    // If exponent is non-zero and also not max, and the implicit bit is 1,
    // then the number is normal value for all values of mantissa.
    bits.set_mantissa(i);
    long double valid = bits;
    ASSERT_EQ(static_cast<int>(isnan(valid)), 0);
    ASSERT_FALSE(bits.is_nan());
  }

  bits.set_biased_exponent(0);
  bits.set_implicit_bit(1);
  for (unsigned int i = 0; i < COUNT; ++i) {
    // If exponent is zero, then the number is a valid but denormal value.
    bits.set_mantissa(i);
    long double valid = bits;
    ASSERT_EQ(static_cast<int>(isnan(valid)), 0);
    ASSERT_FALSE(bits.is_nan());
  }

  bits.set_biased_exponent(0);
  bits.set_implicit_bit(0);
  for (unsigned int i = 0; i < COUNT; ++i) {
    // If exponent is zero, then the number is a valid but denormal value.
    bits.set_mantissa(i);
    long double valid = bits;
    ASSERT_EQ(static_cast<int>(isnan(valid)), 0);
    ASSERT_FALSE(bits.is_nan());
  }
}

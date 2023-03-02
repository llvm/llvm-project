//===-- Unittests for difftime --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/time/difftime.h"
#include "src/time/time_utils.h"
#include "test/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
using __llvm_libc::time_utils::TimeConstants;

TEST(LlvmLibcDifftime, SmokeTest) {
  time_t t1_seconds = TimeConstants::SECONDS_PER_HOUR;
  time_t t2_seconds = 0;

  __llvm_libc::fputil::FPBits<long double> expected_fp =
      __llvm_libc::fputil::FPBits<long double>();
  expected_fp = __llvm_libc::fputil::FPBits<long double>(
      static_cast<long double>(t1_seconds));

  double result = __llvm_libc::difftime(t1_seconds, t2_seconds);

  __llvm_libc::fputil::FPBits<long double> actual_fp =
      __llvm_libc::fputil::FPBits<long double>();
  actual_fp = __llvm_libc::fputil::FPBits<long double>(
      static_cast<long double>(result));

  EXPECT_EQ(actual_fp.bits, expected_fp.bits);
  EXPECT_EQ(actual_fp.get_sign(), expected_fp.get_sign());
  EXPECT_EQ(actual_fp.get_exponent(), expected_fp.get_exponent());
  EXPECT_EQ(actual_fp.get_mantissa(), expected_fp.get_mantissa());
}

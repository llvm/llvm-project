//===-- Unittests for atof ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/stdlib/atof.h"

#include "utils/UnitTest/Test.h"

#include <errno.h>
#include <limits.h>
#include <stddef.h>

// This is just a simple test to make sure that this function works at all. It's
// functionally identical to strtod so the bulk of the testing is there.
TEST(LlvmLibcAToFTest, SimpleTest) {
  __llvm_libc::fputil::FPBits<double> expectedFP =
      __llvm_libc::fputil::FPBits<double>(uint64_t(0x405ec00000000000));

  errno = 0;
  double result = __llvm_libc::atof("123");

  __llvm_libc::fputil::FPBits<double> actualFP =
      __llvm_libc::fputil::FPBits<double>(result);

  EXPECT_EQ(actualFP.bits, expectedFP.bits);
  EXPECT_EQ(actualFP.get_sign(), expectedFP.get_sign());
  EXPECT_EQ(actualFP.get_exponent(), expectedFP.get_exponent());
  EXPECT_EQ(actualFP.get_mantissa(), expectedFP.get_mantissa());
  EXPECT_EQ(errno, 0);
}

TEST(LlvmLibcAToFTest, FailedParsingTest) {
  __llvm_libc::fputil::FPBits<double> expectedFP =
      __llvm_libc::fputil::FPBits<double>(uint64_t(0));

  errno = 0;
  double result = __llvm_libc::atof("???");

  __llvm_libc::fputil::FPBits<double> actualFP =
      __llvm_libc::fputil::FPBits<double>(result);

  EXPECT_EQ(actualFP.bits, expectedFP.bits);
  EXPECT_EQ(actualFP.get_sign(), expectedFP.get_sign());
  EXPECT_EQ(actualFP.get_exponent(), expectedFP.get_exponent());
  EXPECT_EQ(actualFP.get_mantissa(), expectedFP.get_mantissa());
  EXPECT_EQ(errno, 0);
}

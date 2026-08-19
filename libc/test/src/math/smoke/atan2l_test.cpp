//===-- Unittests for atan2l ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atan2l.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAtan2lTest = LIBC_NAMESPACE::testing::FPTest<long double>;

TEST_F(LlvmLibcAtan2lTest, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atan2l(aNaN, zero));
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atan2l(1.0, aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::atan2l(zero, zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::atan2l(neg_zero, zero));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::atan2l(1.0, inf));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::atan2l(-1.0, inf));

  // We're not validating non-trivial test cases here, since values
  // like M_PI may be represented differently depending on the varying
  // size of long double on different platforms.
}

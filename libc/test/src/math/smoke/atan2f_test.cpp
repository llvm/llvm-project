//===-- Unittests for atan2f ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/atan2f.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAtan2fTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcAtan2fTest, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  // TODO: Strengthen errno,exception checks and remove these assert macros
  // after new matchers/test fixtures are added see:
  // https://github.com/llvm/llvm-project/issues/90653.
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atan2f(aNaN, zero));
  // TODO: Uncomment these checks later, RoundingMode affects running
  // tests in this way https://github.com/llvm/llvm-project/issues/90653.
  // EXPECT_FP_EXCEPTION(0);
  EXPECT_MATH_ERRNO(0);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::atan2f(1.0f, aNaN));
  // See above TODO
  // EXPECT_FP_EXCEPTION(0);
  EXPECT_MATH_ERRNO(0);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(0.0f, LIBC_NAMESPACE::atan2f(zero, zero));
  // See above TODO
  // EXPECT_FP_EXCEPTION(0);
  EXPECT_MATH_ERRNO(0);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(-0.0f, LIBC_NAMESPACE::atan2f(-0.0f, zero));
  // See above TODO
  // EXPECT_FP_EXCEPTION(0);
  EXPECT_MATH_ERRNO(0);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(0.0f, LIBC_NAMESPACE::atan2f(1.0f, inf));
  // See above TODO
  // EXPECT_FP_EXCEPTION(0);
  EXPECT_MATH_ERRNO(0);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  EXPECT_FP_EQ_ALL_ROUNDING(-0.0f, LIBC_NAMESPACE::atan2f(-1.0f, inf));
  // See above TODO
  // EXPECT_FP_EXCEPTION(0);
  EXPECT_MATH_ERRNO(0);
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcAtan2fTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(0x1.921fb6p-1f,
               LIBC_NAMESPACE::atan2f(min_denormal, min_denormal));
  EXPECT_FP_EQ(0x1.000002p-23f,
               LIBC_NAMESPACE::atan2f(min_denormal, max_denormal));
  EXPECT_FP_EQ(0x1.921fb4p0f,
               LIBC_NAMESPACE::atan2f(max_denormal, min_denormal));
  EXPECT_FP_EQ(0x1.921fb6p-1f,
               LIBC_NAMESPACE::atan2f(max_denormal, max_denormal));
}

TEST_F(LlvmLibcAtan2fTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::atan2f(min_denormal, min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::atan2f(min_denormal, max_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::atan2f(max_denormal, min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::atan2f(max_denormal, max_denormal));
}

TEST_F(LlvmLibcAtan2fTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::atan2f(min_denormal, min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::atan2f(min_denormal, max_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::atan2f(max_denormal, min_denormal));
  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::atan2f(max_denormal, max_denormal));
}

#endif

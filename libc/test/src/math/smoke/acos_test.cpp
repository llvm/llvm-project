//===-- Unittests for acos ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "src/__support/libc_errno.h"
#include "src/math/acos.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcAcosTest = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcAcosTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acos(sNaN),
                                           FE_INVALID);
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acos(aNaN));
  EXPECT_FP_EQ(0x1.921fb54442d18p0, LIBC_NAMESPACE::acos(zero));
  EXPECT_FP_EQ(0x1.921fb54442d18p0, LIBC_NAMESPACE::acos(neg_zero));

  libc_errno = 0;
  EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acos(inf),
                                           FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
  EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acos(neg_inf),
                                           FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
  EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acos(2.0),
                                           FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
  EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acos(-2.0),
                                           FE_INVALID);
  EXPECT_MATH_ERRNO(EDOM);
  EXPECT_FP_EQ(zero, LIBC_NAMESPACE::acos(1.0));
  EXPECT_FP_EQ(0x1.921fb54442d18p1, LIBC_NAMESPACE::acos(-1.0));
  EXPECT_FP_EQ(0x1.921fb54442d18p0, LIBC_NAMESPACE::acos(0x1.0p-54));
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcAcosTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(0x1.921fb54442d18p0, LIBC_NAMESPACE::acos(min_denormal));
}

TEST_F(LlvmLibcAcosTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(0x1.921fb54442d18p0, LIBC_NAMESPACE::acos(min_denormal));
}

TEST_F(LlvmLibcAcosTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(0x1.921fb54442d18p0, LIBC_NAMESPACE::acos(min_denormal));
}

#endif

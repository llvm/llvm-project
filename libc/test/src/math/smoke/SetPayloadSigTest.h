//===-- Utility class to test flavors of setpayloadsig ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_TEST_SRC_MATH_SMOKE_SETPAYLOADSIGTEST_H
#define LIBC_TEST_SRC_MATH_SMOKE_SETPAYLOADSIGTEST_H

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T>
class SetPayloadSigTestTemplate : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef int (*SetPayloadSigFunc)(T *, T);

  void testInvalidPayloads(SetPayloadSigFunc func) {
    T res;

    EXPECT_EQ(1, func(&res, T(aNaN)));
    EXPECT_EQ(1, func(&res, T(neg_aNaN)));
    EXPECT_EQ(1, func(&res, T(inf)));
    EXPECT_EQ(1, func(&res, T(neg_inf)));
    EXPECT_EQ(1, func(&res, T(0.0)));
    EXPECT_EQ(1, func(&res, T(-0.0)));
    EXPECT_EQ(1, func(&res, T(0.1)));
    EXPECT_EQ(1, func(&res, T(-0.1)));
    EXPECT_EQ(1, func(&res, T(-1.0)));
    EXPECT_EQ(1, func(&res, T(0x42.1p+0)));
    EXPECT_EQ(1, func(&res, T(-0x42.1p+0)));
    EXPECT_EQ(1, func(&res, T(StorageType(1) << (FPBits::FRACTION_LEN - 1))));
  }

  void testValidPayloads(SetPayloadSigFunc func) {
    T res;

    EXPECT_EQ(0, func(&res, T(1.0)));
    EXPECT_TRUE(FPBits(res).is_signaling_nan());
    EXPECT_EQ(FPBits::signaling_nan(Sign::POS, 1).uintval(),
              FPBits(res).uintval());

    EXPECT_EQ(0, func(&res, T(0x42.0p+0)));
    EXPECT_TRUE(FPBits(res).is_signaling_nan());
    EXPECT_EQ(FPBits::signaling_nan(Sign::POS, 0x42).uintval(),
              FPBits(res).uintval());

    EXPECT_EQ(0, func(&res, T(0x123.0p+0)));
    EXPECT_TRUE(FPBits(res).is_signaling_nan());
    EXPECT_EQ(FPBits::signaling_nan(Sign::POS, 0x123).uintval(),
              FPBits(res).uintval());

    EXPECT_EQ(0, func(&res, T(FPBits::FRACTION_MASK >> 1)));
    EXPECT_TRUE(FPBits(res).is_signaling_nan());
    EXPECT_EQ(
        FPBits::signaling_nan(Sign::POS, FPBits::FRACTION_MASK >> 1).uintval(),
        FPBits(res).uintval());
  }
};

#define LIST_SETPAYLOADSIG_TESTS(T, func)                                      \
  using LlvmLibcSetPayloadSigTest = SetPayloadSigTestTemplate<T>;              \
  TEST_F(LlvmLibcSetPayloadSigTest, InvalidPayloads) {                         \
    testInvalidPayloads(&func);                                                \
  }                                                                            \
  TEST_F(LlvmLibcSetPayloadSigTest, ValidPayloads) { testValidPayloads(&func); }

#endif // LIBC_TEST_SRC_MATH_SMOKE_SETPAYLOADSIGTEST_H

//===-- Utility class to test fxbits -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"
#include "src/__support/fixed_point/fx_rep.h"

template <typename T, typename XType> class FxbitsTest : public LIBC_NAMESPACE::testing::Test {
    using FXRep = LIBC_NAMESPACE::fixed_point::FXRep<T>;
    static constexpr T zero = FXRep::ZERO();
    static constexpr T min = FXRep::MIN();
    static constexpr T max = FXRep::MAX();
    static constexpr T half = static_cast<T>(0.5);
    static constexpr T neg_half = static_cast<T>(-0.5);
    static constexpr T one =
        (FXRep::INTEGRAL_LEN > 0) ? static_cast<T>(1) : FXRep::MAX();
    static constexpr T neg_one = static_cast<T>(-1);
    static constexpr T eps = FXRep::EPS();

public:
    typedef T (*FxbitsFunc)(XType); 

    void testSpecialNumbers(FxbitsFunc func) {
        EXPECT_EQ(zero, func(0));
        EXPECT_EQ(half, func((XType) (0b1 << (FXRep::FRACTION_LEN - 1)))); // 0.1000...b
        EXPECT_EQ(min, func((XType) 0x1));
        EXPECT_EQ(one, func(1));
        EXPECT_EQ(neg_one, func(-1));
    }
};


#define LIST_FXBITS_TEST(T, XType, func)                                        \
    using LlvmLibcFxbitsTest = FxbitsTest<T, XType>;                            \
    TEST_F(LlvmLibcFxbitsTest, SpecialNumbers) { testSpecialNumbers(&func); }   \
    static_assert(true, "Require semicolon.")

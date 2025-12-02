//===-- Utility class to test fxdivi functions ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/type_traits.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/fixed_point/fx_rep.h"
#include "test/UnitTest/Test.h"

template <typename XType> XType get_epsilon() = delete;
template <> fract get_epsilon() { return FRACT_EPSILON; }
template <> unsigned fract get_epsilon() { return UFRACT_EPSILON; }
template <> long fract get_epsilon() { return LFRACT_EPSILON; }

template <typename XType>
class DivITest : public LIBC_NAMESPACE::testing::Test {
  using FXRep = LIBC_NAMESPACE::fixed_point::FXRep<XType>;
  using FXBits = LIBC_NAMESPACE::fixed_point::FXBits<XType>;

public:
  typedef XType (*DivIFunc)(int, int);

  void testBasic(DivIFunc func) {
    XType epsilon = get_epsilon<XType>();
    EXPECT_LT((func(2, 3) - 0.666656494140625r), epsilon);
    EXPECT_LT((func(3, 4) - 0.75r), epsilon);
    EXPECT_LT((func(1043, 2764) - 0.3773516643r), epsilon);
    EXPECT_LT((func(60000, 720293) - 0.08329943509r), epsilon);

    EXPECT_EQ(func(128, 256), 0.5r);
    EXPECT_EQ(func(1, 2), 0.5r);
    EXPECT_EQ(func(1, 4), 0.25r);
    EXPECT_EQ(func(1, 8), 0.125r);
    EXPECT_EQ(func(1, 16), 0.0625r);

    EXPECT_EQ(func(-1, 2), -0.5r);
    EXPECT_EQ(func(1, -4), -0.25r);
    EXPECT_EQ(func(-1, 8), -0.125r);
    EXPECT_EQ(func(1, -16), -0.0625r);
  }

  void testSpecial(DivIFunc func) {
    XType epsilon = get_epsilon<XType>();
    EXPECT_EQ(func(0, 10), 0.r);
    EXPECT_EQ(func(0, -10), 0.r);
    EXPECT_EQ(func(-(1 << FRACT_FBIT), 1 << FRACT_FBIT), FRACT_MIN);
    EXPECT_EQ(func((1 << FRACT_FBIT) - 1, 1 << FRACT_FBIT), FRACT_MAX);
    // From Section 7.18a.6.1, functions returning a fixed-point value, the
    // return value is saturated on overflow.
    EXPECT_EQ(func(INT_MAX, INT_MAX), FRACT_MAX);
    EXPECT_LT(func(INT_MAX - 1, INT_MAX) - 0.99999999r, epsilon);
    EXPECT_EQ(func(INT_MIN, INT_MAX), FRACT_MIN);
    // Expecting 0 here as fract is not precise enough to
    // handle 1/INT_MAX
    EXPECT_LT(func(1, INT_MAX) - 0.r, epsilon);
    // This results in 1.1739, which should be saturated to FRACT_MAX
    EXPECT_EQ(func(27, 23), FRACT_MAX);

    EXPECT_EQ(func(INT_MIN, 1), FRACT_MIN);
    EXPECT_LT(func(1, INT_MIN) - 0.r, epsilon);

    EXPECT_EQ(func(INT_MIN, INT_MIN), 1.r);
  }
};

#define LIST_DIVI_TESTS(Name, XType, func)                                     \
  using LlvmLibc##Name##diviTest = DivITest<XType>;                            \
  TEST_F(LlvmLibc##Name##diviTest, Basic) { testBasic(&func); }                \
  TEST_F(LlvmLibc##Name##diviTest, Special) { testSpecial(&func); }            \
  static_assert(true, "Require semicolon.")

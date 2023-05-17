//===-- Utility class to test fmod special numbers ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_FMODTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_FMODTEST_H

#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/NearestIntegerOperations.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <limits>
#include <math.h>

#define TEST_SPECIAL(x, y, expected, dom_err, expected_exception)              \
  EXPECT_FP_EQ(expected, f(x, y));                                             \
  EXPECT_MATH_ERRNO((dom_err) ? EDOM : 0);                                     \
  EXPECT_FP_EXCEPTION(expected_exception);                                     \
  __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT)

#define TEST_REGULAR(x, y, expected) TEST_SPECIAL(x, y, expected, false, 0)

template <typename T> class FmodTest : public __llvm_libc::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*FModFunc)(T, T);

  void testSpecialNumbers(FModFunc f) {
    using nl = std::numeric_limits<T>;

    // fmod (+0, y) == +0 for y != 0.
    TEST_SPECIAL(0.0, 3.0, 0.0, false, 0);
    TEST_SPECIAL(0.0, nl::denorm_min(), 0.0, false, 0);
    TEST_SPECIAL(0.0, -nl::denorm_min(), 0.0, false, 0);
    TEST_SPECIAL(0.0, nl::min(), 0.0, false, 0);
    TEST_SPECIAL(0.0, -nl::min(), 0.0, false, 0);
    TEST_SPECIAL(0.0, nl::max(), 0.0, false, 0);
    TEST_SPECIAL(0.0, -nl::max(), 0.0, false, 0);

    // fmod (-0, y) == -0 for y != 0.
    TEST_SPECIAL(neg_zero, 3.0, neg_zero, false, 0);
    TEST_SPECIAL(neg_zero, nl::denorm_min(), neg_zero, false, 0);
    TEST_SPECIAL(neg_zero, -nl::denorm_min(), neg_zero, false, 0);
    TEST_SPECIAL(neg_zero, nl::min(), neg_zero, false, 0);
    TEST_SPECIAL(neg_zero, -nl::min(), neg_zero, false, 0);
    TEST_SPECIAL(neg_zero, nl::max(), neg_zero, false, 0);
    TEST_SPECIAL(neg_zero, -nl::max(), neg_zero, false, 0);

    // fmod (+inf, y) == nl::quiet_NaN() plus invalid exception.
    TEST_SPECIAL(inf, 3.0, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(inf, -1.1L, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(inf, 0.0, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(inf, neg_zero, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(inf, nl::denorm_min(), nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(inf, nl::min(), nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(inf, nl::max(), nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(inf, inf, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(inf, neg_inf, nl::quiet_NaN(), true, FE_INVALID);

    // fmod (-inf, y) == nl::quiet_NaN() plus invalid exception.
    TEST_SPECIAL(neg_inf, 3.0, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(neg_inf, -1.1L, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(neg_inf, 0.0, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(neg_inf, neg_zero, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(neg_inf, nl::denorm_min(), nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(neg_inf, nl::min(), nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(neg_inf, nl::max(), nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(neg_inf, inf, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(neg_inf, neg_inf, nl::quiet_NaN(), true, FE_INVALID);

    // fmod (x, +0) == nl::quiet_NaN() plus invalid exception.
    TEST_SPECIAL(3.0, 0.0, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(-1.1L, 0.0, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(0.0, 0.0, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(neg_zero, 0.0, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(nl::denorm_min(), 0.0, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(nl::min(), 0.0, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(nl::max(), 0.0, nl::quiet_NaN(), true, FE_INVALID);

    // fmod (x, -0) == nl::quiet_NaN() plus invalid exception.
    TEST_SPECIAL(3.0, neg_zero, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(-1.1L, neg_zero, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(0.0, neg_zero, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(neg_zero, neg_zero, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(nl::denorm_min(), neg_zero, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(nl::min(), neg_zero, nl::quiet_NaN(), true, FE_INVALID);
    TEST_SPECIAL(nl::max(), neg_zero, nl::quiet_NaN(), true, FE_INVALID);

    // fmod (x, +inf) == x for x not infinite.
    TEST_SPECIAL(0.0, inf, 0.0, false, 0);
    TEST_SPECIAL(neg_zero, inf, neg_zero, false, 0);
    TEST_SPECIAL(nl::denorm_min(), inf, nl::denorm_min(), false, 0);
    TEST_SPECIAL(nl::min(), inf, nl::min(), false, 0);
    TEST_SPECIAL(nl::max(), inf, nl::max(), false, 0);
    TEST_SPECIAL(3.0, inf, 3.0, false, 0);
    // fmod (x, -inf) == x for x not infinite.
    TEST_SPECIAL(0.0, neg_inf, 0.0, false, 0);
    TEST_SPECIAL(neg_zero, neg_inf, neg_zero, false, 0);
    TEST_SPECIAL(nl::denorm_min(), neg_inf, nl::denorm_min(), false, 0);
    TEST_SPECIAL(nl::min(), neg_inf, nl::min(), false, 0);
    TEST_SPECIAL(nl::max(), neg_inf, nl::max(), false, 0);
    TEST_SPECIAL(3.0, neg_inf, 3.0, false, 0);

    TEST_SPECIAL(0.0, nl::quiet_NaN(), nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(0.0, -nl::quiet_NaN(), nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(neg_zero, nl::quiet_NaN(), nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(neg_zero, -nl::quiet_NaN(), nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(1.0, nl::quiet_NaN(), nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(1.0, -nl::quiet_NaN(), nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(inf, nl::quiet_NaN(), nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(inf, -nl::quiet_NaN(), nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(neg_inf, nl::quiet_NaN(), nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(neg_inf, -nl::quiet_NaN(), nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(0.0, nl::signaling_NaN(), nl::quiet_NaN(), false, FE_INVALID);
    TEST_SPECIAL(0.0, -nl::signaling_NaN(), nl::quiet_NaN(), false, FE_INVALID);
    TEST_SPECIAL(neg_zero, nl::signaling_NaN(), nl::quiet_NaN(), false,
                 FE_INVALID);
    TEST_SPECIAL(neg_zero, -nl::signaling_NaN(), nl::quiet_NaN(), false,
                 FE_INVALID);
    TEST_SPECIAL(1.0, nl::signaling_NaN(), nl::quiet_NaN(), false, FE_INVALID);
    TEST_SPECIAL(1.0, -nl::signaling_NaN(), nl::quiet_NaN(), false, FE_INVALID);
    TEST_SPECIAL(inf, nl::signaling_NaN(), nl::quiet_NaN(), false, FE_INVALID);
    TEST_SPECIAL(inf, -nl::signaling_NaN(), nl::quiet_NaN(), false, FE_INVALID);
    TEST_SPECIAL(neg_inf, nl::signaling_NaN(), nl::quiet_NaN(), false,
                 FE_INVALID);
    TEST_SPECIAL(neg_inf, -nl::signaling_NaN(), nl::quiet_NaN(), false,
                 FE_INVALID);
    TEST_SPECIAL(nl::quiet_NaN(), 0.0, nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(-nl::quiet_NaN(), 0.0, nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(nl::quiet_NaN(), neg_zero, nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(-nl::quiet_NaN(), neg_zero, nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(nl::quiet_NaN(), 1.0, nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(-nl::quiet_NaN(), 1.0, nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(nl::quiet_NaN(), inf, nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(-nl::quiet_NaN(), inf, nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(nl::quiet_NaN(), neg_inf, nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(-nl::quiet_NaN(), neg_inf, nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(nl::signaling_NaN(), 0.0, nl::quiet_NaN(), false, FE_INVALID);
    TEST_SPECIAL(-nl::signaling_NaN(), 0.0, nl::quiet_NaN(), false, FE_INVALID);
    TEST_SPECIAL(nl::signaling_NaN(), neg_zero, nl::quiet_NaN(), false,
                 FE_INVALID);
    TEST_SPECIAL(-nl::signaling_NaN(), neg_zero, nl::quiet_NaN(), false,
                 FE_INVALID);
    TEST_SPECIAL(nl::signaling_NaN(), 1.0, nl::quiet_NaN(), false, FE_INVALID);
    TEST_SPECIAL(-nl::signaling_NaN(), 1.0, nl::quiet_NaN(), false, FE_INVALID);
    TEST_SPECIAL(nl::signaling_NaN(), inf, nl::quiet_NaN(), false, FE_INVALID);
    TEST_SPECIAL(-nl::signaling_NaN(), inf, nl::quiet_NaN(), false, FE_INVALID);
    TEST_SPECIAL(nl::signaling_NaN(), neg_inf, nl::quiet_NaN(), false,
                 FE_INVALID);
    TEST_SPECIAL(-nl::signaling_NaN(), neg_inf, nl::quiet_NaN(), false,
                 FE_INVALID);
    TEST_SPECIAL(nl::quiet_NaN(), nl::quiet_NaN(), nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(nl::quiet_NaN(), -nl::quiet_NaN(), nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(-nl::quiet_NaN(), nl::quiet_NaN(), nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(-nl::quiet_NaN(), -nl::quiet_NaN(), nl::quiet_NaN(), false, 0);
    TEST_SPECIAL(nl::quiet_NaN(), nl::signaling_NaN(), nl::quiet_NaN(), false,
                 FE_INVALID);
    TEST_SPECIAL(nl::quiet_NaN(), -nl::signaling_NaN(), nl::quiet_NaN(), false,
                 FE_INVALID);
    TEST_SPECIAL(-nl::quiet_NaN(), nl::signaling_NaN(), nl::quiet_NaN(), false,
                 FE_INVALID);
    TEST_SPECIAL(-nl::quiet_NaN(), -nl::signaling_NaN(), nl::quiet_NaN(), false,
                 FE_INVALID);
    TEST_SPECIAL(nl::signaling_NaN(), nl::quiet_NaN(), nl::quiet_NaN(), false,
                 FE_INVALID);
    TEST_SPECIAL(nl::signaling_NaN(), -nl::quiet_NaN(), nl::quiet_NaN(), false,
                 FE_INVALID);
    TEST_SPECIAL(-nl::signaling_NaN(), nl::quiet_NaN(), nl::quiet_NaN(), false,
                 FE_INVALID);
    TEST_SPECIAL(-nl::signaling_NaN(), -nl::quiet_NaN(), nl::quiet_NaN(), false,
                 FE_INVALID);
    TEST_SPECIAL(nl::signaling_NaN(), nl::signaling_NaN(), nl::quiet_NaN(),
                 false, FE_INVALID);
    TEST_SPECIAL(nl::signaling_NaN(), -nl::signaling_NaN(), nl::quiet_NaN(),
                 false, FE_INVALID);
    TEST_SPECIAL(-nl::signaling_NaN(), nl::signaling_NaN(), nl::quiet_NaN(),
                 false, FE_INVALID);
    TEST_SPECIAL(-nl::signaling_NaN(), -nl::signaling_NaN(), nl::quiet_NaN(),
                 false, FE_INVALID);

    TEST_SPECIAL(6.5, 2.25L, 2.0L, false, 0);
    TEST_SPECIAL(-6.5, 2.25L, -2.0L, false, 0);
    TEST_SPECIAL(6.5, -2.25L, 2.0L, false, 0);
    TEST_SPECIAL(-6.5, -2.25L, -2.0L, false, 0);

    TEST_SPECIAL(nl::max(), nl::max(), 0.0, false, 0);
    TEST_SPECIAL(nl::max(), -nl::max(), 0.0, false, 0);
    TEST_SPECIAL(nl::max(), nl::min(), 0.0, false, 0);
    TEST_SPECIAL(nl::max(), -nl::min(), 0.0, false, 0);
    TEST_SPECIAL(nl::max(), nl::denorm_min(), 0.0, false, 0);
    TEST_SPECIAL(nl::max(), -nl::denorm_min(), 0.0, false, 0);
    TEST_SPECIAL(-nl::max(), nl::max(), neg_zero, false, 0);
    TEST_SPECIAL(-nl::max(), -nl::max(), neg_zero, false, 0);
    TEST_SPECIAL(-nl::max(), nl::min(), neg_zero, false, 0);
    TEST_SPECIAL(-nl::max(), -nl::min(), neg_zero, false, 0);
    TEST_SPECIAL(-nl::max(), nl::denorm_min(), neg_zero, false, 0);
    TEST_SPECIAL(-nl::max(), -nl::denorm_min(), neg_zero, false, 0);

    TEST_SPECIAL(nl::min(), nl::max(), nl::min(), false, 0);
    TEST_SPECIAL(nl::min(), -nl::max(), nl::min(), false, 0);
    TEST_SPECIAL(nl::min(), nl::min(), 0.0, false, 0);
    TEST_SPECIAL(nl::min(), -nl::min(), 0.0, false, 0);
    TEST_SPECIAL(nl::min(), nl::denorm_min(), 0.0, false, 0);
    TEST_SPECIAL(nl::min(), -nl::denorm_min(), 0.0, false, 0);
    TEST_SPECIAL(-nl::min(), nl::max(), -nl::min(), false, 0);
    TEST_SPECIAL(-nl::min(), -nl::max(), -nl::min(), false, 0);
    TEST_SPECIAL(-nl::min(), nl::min(), neg_zero, false, 0);
    TEST_SPECIAL(-nl::min(), -nl::min(), neg_zero, false, 0);
    TEST_SPECIAL(-nl::min(), nl::denorm_min(), neg_zero, false, 0);
    TEST_SPECIAL(-nl::min(), -nl::denorm_min(), neg_zero, false, 0);

    TEST_SPECIAL(nl::denorm_min(), nl::max(), nl::denorm_min(), false, 0);
    TEST_SPECIAL(nl::denorm_min(), -nl::max(), nl::denorm_min(), false, 0);
    TEST_SPECIAL(nl::denorm_min(), nl::min(), nl::denorm_min(), false, 0);
    TEST_SPECIAL(nl::denorm_min(), -nl::min(), nl::denorm_min(), false, 0);
    TEST_SPECIAL(nl::denorm_min(), nl::denorm_min(), 0.0, false, 0);
    TEST_SPECIAL(nl::denorm_min(), -nl::denorm_min(), 0.0, false, 0);
    TEST_SPECIAL(-nl::denorm_min(), nl::max(), -nl::denorm_min(), false, 0);
    TEST_SPECIAL(-nl::denorm_min(), -nl::max(), -nl::denorm_min(), false, 0);
    TEST_SPECIAL(-nl::denorm_min(), nl::min(), -nl::denorm_min(), false, 0);
    TEST_SPECIAL(-nl::denorm_min(), -nl::min(), -nl::denorm_min(), false, 0);
    TEST_SPECIAL(-nl::denorm_min(), nl::denorm_min(), neg_zero, false, 0);
    TEST_SPECIAL(-nl::denorm_min(), -nl::denorm_min(), neg_zero, false, 0);
  }

  void testRegularExtreme(FModFunc f) {

    TEST_REGULAR(0x1p127L, 0x3p-149L, 0x1p-149L);
    TEST_REGULAR(0x1p127L, -0x3p-149L, 0x1p-149L);
    TEST_REGULAR(0x1p127L, 0x3p-148L, 0x1p-147L);
    TEST_REGULAR(0x1p127L, -0x3p-148L, 0x1p-147L);
    TEST_REGULAR(0x1p127L, 0x3p-126L, 0x1p-125L);
    TEST_REGULAR(0x1p127L, -0x3p-126L, 0x1p-125L);
    TEST_REGULAR(-0x1p127L, 0x3p-149L, -0x1p-149L);
    TEST_REGULAR(-0x1p127L, -0x3p-149L, -0x1p-149L);
    TEST_REGULAR(-0x1p127L, 0x3p-148L, -0x1p-147L);
    TEST_REGULAR(-0x1p127L, -0x3p-148L, -0x1p-147L);
    TEST_REGULAR(-0x1p127L, 0x3p-126L, -0x1p-125L);
    TEST_REGULAR(-0x1p127L, -0x3p-126L, -0x1p-125L);

    if constexpr (sizeof(T) >= sizeof(double)) {
      TEST_REGULAR(0x1p1023L, 0x3p-1074L, 0x1p-1073L);
      TEST_REGULAR(0x1p1023L, -0x3p-1074L, 0x1p-1073L);
      TEST_REGULAR(0x1p1023L, 0x3p-1073L, 0x1p-1073L);
      TEST_REGULAR(0x1p1023L, -0x3p-1073L, 0x1p-1073L);
      TEST_REGULAR(0x1p1023L, 0x3p-1022L, 0x1p-1021L);
      TEST_REGULAR(0x1p1023L, -0x3p-1022L, 0x1p-1021L);
      TEST_REGULAR(-0x1p1023L, 0x3p-1074L, -0x1p-1073L);
      TEST_REGULAR(-0x1p1023L, -0x3p-1074L, -0x1p-1073L);
      TEST_REGULAR(-0x1p1023L, 0x3p-1073L, -0x1p-1073L);
      TEST_REGULAR(-0x1p1023L, -0x3p-1073L, -0x1p-1073L);
      TEST_REGULAR(-0x1p1023L, 0x3p-1022L, -0x1p-1021L);
      TEST_REGULAR(-0x1p1023L, -0x3p-1022L, -0x1p-1021L);
    }
  }
};

#define LIST_FMOD_TESTS(T, func)                                               \
  using LlvmLibcFmodTest = FmodTest<T>;                                        \
  TEST_F(LlvmLibcFmodTest, SpecialNumbers) { testSpecialNumbers(&func); }      \
  TEST_F(LlvmLibcFmodTest, RegularExtreme) { testRegularExtreme(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_FMODTEST_H

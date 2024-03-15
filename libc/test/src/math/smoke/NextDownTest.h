//===-- Utility class to test different flavors of nextdown -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_NEXTDOWNTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_NEXTDOWNTEST_H

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T>
class NextDownTestTemplate : public LIBC_NAMESPACE::testing::Test {
  using FPBits = typename LIBC_NAMESPACE::fputil::FPBits<T>;
  using Sign = typename LIBC_NAMESPACE::fputil::Sign;

  static constexpr T inf = FPBits::inf().get_val();
  static constexpr T neg_inf = FPBits::inf(Sign::NEG).get_val();
  static constexpr T zero = FPBits::zero().get_val();
  static constexpr T neg_zero = FPBits::zero(Sign::NEG).get_val();
  static constexpr T nan = FPBits::quiet_nan().get_val();

  static constexpr T min_subnormal = FPBits::min_subnormal().get_val();
  static constexpr T neg_min_subnormal =
      FPBits::min_subnormal(Sign::NEG).get_val();
  static constexpr T max_normal = FPBits::max_normal().get_val();
  static constexpr T neg_max_normal = FPBits::max_normal(Sign::NEG).get_val();

public:
  typedef T (*NextDownFunc)(T);

  void testNaN(NextDownFunc func) { ASSERT_FP_EQ(func(nan), nan); }

  void testBoundaries(NextDownFunc func) {
    ASSERT_FP_EQ(zero, func(min_subnormal));

    ASSERT_FP_EQ(neg_min_subnormal, func(zero));
    ASSERT_FP_EQ(neg_min_subnormal, func(neg_zero));

    ASSERT_FP_EQ(neg_max_normal, func(neg_max_normal));
    ASSERT_FP_EQ(neg_inf, func(neg_inf));

    ASSERT_FP_EQ(max_normal, func(inf));
  }
};

#define LIST_NEXTDOWN_TESTS(T, func)                                           \
  using LlvmLibcNextDownTest = NextDownTestTemplate<T>;                        \
  TEST_F(LlvmLibcNextDownTest, TestNaN) { testNaN(&func); }                    \
  TEST_F(LlvmLibcNextDownTest, TestBoundaries) { testBoundaries(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_NEXTDOWNTEST_H

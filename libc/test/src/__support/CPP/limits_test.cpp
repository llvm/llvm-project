//===-- Unittests for Limits ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/Limits.h"
#include "utils/UnitTest/Test.h"

// This just checks against the C spec, almost all implementations will surpass
// this.
TEST(LlvmLibcLimitsTest, LimitsFollowSpec) {
  ASSERT_EQ(__llvm_libc::cpp::NumericLimits<int>::max(), INT_MAX);
  ASSERT_EQ(__llvm_libc::cpp::NumericLimits<int>::min(), INT_MIN);

  ASSERT_EQ(__llvm_libc::cpp::NumericLimits<unsigned int>::max(), UINT_MAX);

  ASSERT_EQ(__llvm_libc::cpp::NumericLimits<long>::max(), LONG_MAX);
  ASSERT_EQ(__llvm_libc::cpp::NumericLimits<long>::min(), LONG_MIN);

  ASSERT_EQ(__llvm_libc::cpp::NumericLimits<unsigned long>::max(), ULONG_MAX);

  ASSERT_EQ(__llvm_libc::cpp::NumericLimits<long long>::max(), LLONG_MAX);
  ASSERT_EQ(__llvm_libc::cpp::NumericLimits<long long>::min(), LLONG_MIN);

  ASSERT_EQ(__llvm_libc::cpp::NumericLimits<unsigned long long>::max(),
            ULLONG_MAX);
}

#ifdef __SIZEOF_INT128__
// This checks that the current environment supports 128 bit integers.
TEST(LlvmLibcLimitsTest, Int128Works) {
  __int128_t max128 = ~__uint128_t(0) >> 1;
  __int128_t min128 = (__int128_t(1) << 127);
  EXPECT_GT(__llvm_libc::cpp::NumericLimits<__int128_t>::max(),
            __int128_t(__llvm_libc::cpp::NumericLimits<long long>::max()));
  ASSERT_EQ(__llvm_libc::cpp::NumericLimits<__int128_t>::max(), max128);

  EXPECT_LT(__llvm_libc::cpp::NumericLimits<__int128_t>::min(),
            __int128_t(__llvm_libc::cpp::NumericLimits<long long>::min()));
  ASSERT_EQ(__llvm_libc::cpp::NumericLimits<__int128_t>::min(), min128);

  __uint128_t umax128 = ~__uint128_t(0);
  EXPECT_GT(
      __llvm_libc::cpp::NumericLimits<__uint128_t>::max(),
      __uint128_t(__llvm_libc::cpp::NumericLimits<unsigned long long>::max()));
  ASSERT_EQ(__llvm_libc::cpp::NumericLimits<__uint128_t>::max(), umax128);
}
#endif

//===-- Unittests for Limits ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/Limits.h"
#include "src/__support/CPP/UInt.h"
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

TEST(LlvmLibcLimitsTest, UInt128Limits) {
  auto umax128 =
      __llvm_libc::cpp::NumericLimits<__llvm_libc::cpp::UInt<128>>::max();
  auto umax64 = __llvm_libc::cpp::UInt<128>(
      __llvm_libc::cpp::NumericLimits<uint64_t>::max());
  EXPECT_GT(umax128, umax64);
  ASSERT_EQ(~__llvm_libc::cpp::UInt<128>(0), umax128);
#ifdef __SIZEOF_INT128__
  ASSERT_EQ(~__uint128_t(0),
            __llvm_libc::cpp::NumericLimits<__uint128_t>::max());
#endif
}

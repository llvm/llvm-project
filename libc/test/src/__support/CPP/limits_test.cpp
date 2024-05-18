//===-- Unittests for Limits ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/__support/big_int.h"
#include "src/__support/macros/properties/types.h" // LIBC_TYPES_HAS_INT128
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE {

// This just checks against the C spec, almost all implementations will surpass
// this.
TEST(LlvmLibcLimitsTest, LimitsFollowSpec) {
  ASSERT_EQ(cpp::numeric_limits<int>::max(), INT_MAX);
  ASSERT_EQ(cpp::numeric_limits<int>::min(), INT_MIN);

  ASSERT_EQ(cpp::numeric_limits<unsigned int>::max(), UINT_MAX);

  ASSERT_EQ(cpp::numeric_limits<long>::max(), LONG_MAX);
  ASSERT_EQ(cpp::numeric_limits<long>::min(), LONG_MIN);

  ASSERT_EQ(cpp::numeric_limits<unsigned long>::max(), ULONG_MAX);

  ASSERT_EQ(cpp::numeric_limits<long long>::max(), LLONG_MAX);
  ASSERT_EQ(cpp::numeric_limits<long long>::min(), LLONG_MIN);

  ASSERT_EQ(cpp::numeric_limits<unsigned long long>::max(), ULLONG_MAX);
}

TEST(LlvmLibcLimitsTest, UInt128Limits) {
  auto umax128 = cpp::numeric_limits<LIBC_NAMESPACE::UInt<128>>::max();
  auto umax64 = LIBC_NAMESPACE::UInt<128>(cpp::numeric_limits<uint64_t>::max());
  EXPECT_GT(umax128, umax64);
  ASSERT_EQ(~LIBC_NAMESPACE::UInt<128>(0), umax128);
#ifdef LIBC_TYPES_HAS_INT128
  ASSERT_EQ(~__uint128_t(0), cpp::numeric_limits<__uint128_t>::max());
#endif // LIBC_TYPES_HAS_INT128
}

} // namespace LIBC_NAMESPACE

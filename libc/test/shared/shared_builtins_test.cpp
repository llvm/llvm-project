//===-- Unittests for shared builtins -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "shared/builtins.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSharedBuiltinsTest, AllFloat) {
  EXPECT_FP_EQ(3.0f, LIBC_NAMESPACE::shared::addsf3(1.0f, 2.0f));
  EXPECT_FP_EQ(3.0f, LIBC_NAMESPACE::shared::divsf3(6.0f, 2.0f));
  EXPECT_FP_EQ(6.0f, LIBC_NAMESPACE::shared::mulsf3(2.0f, 3.0f));
  EXPECT_FP_EQ(2.0f, LIBC_NAMESPACE::shared::subsf3(5.0f, 3.0f));
}

TEST(LlvmLibcSharedBuiltinsTest, AllDouble) {
  EXPECT_FP_EQ(3.0, LIBC_NAMESPACE::shared::adddf3(1.0, 2.0));
  EXPECT_FP_EQ(3.0, LIBC_NAMESPACE::shared::divdf3(6.0, 2.0));
  EXPECT_FP_EQ(6.0, LIBC_NAMESPACE::shared::muldf3(2.0, 3.0));
  EXPECT_FP_EQ(2.0, LIBC_NAMESPACE::shared::subdf3(5.0, 3.0));
}

#ifdef LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedBuiltinsTest, AllFloat128) {
  EXPECT_FP_EQ(float128(3.0),
               LIBC_NAMESPACE::shared::addtf3(float128(1.0), float128(2.0)));
  EXPECT_FP_EQ(float128(3.0),
               LIBC_NAMESPACE::shared::divtf3(float128(6.0), float128(2.0)));
  EXPECT_FP_EQ(float128(6.0),
               LIBC_NAMESPACE::shared::multf3(float128(2.0), float128(3.0)));
  EXPECT_FP_EQ(float128(2.0),
               LIBC_NAMESPACE::shared::subtf3(float128(5.0), float128(3.0)));
}

#endif // LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedBuiltinsTest, FloatToIntConversion) {
  EXPECT_EQ(int64_t(12), LIBC_NAMESPACE::shared::fixsfdi(12.5f));
  EXPECT_EQ(int32_t(12), LIBC_NAMESPACE::shared::fixsfsi(12.5f));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_EQ(static_cast<__int128_t>(12),
            LIBC_NAMESPACE::shared::fixsfti(12.5f));
#endif // LIBC_TYPES_HAS_INT128
  EXPECT_EQ(uint64_t(12), LIBC_NAMESPACE::shared::fixunssfdi(12.5f));
  EXPECT_EQ(uint32_t(12), LIBC_NAMESPACE::shared::fixunssfsi(12.5f));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_EQ(static_cast<__uint128_t>(12),
            LIBC_NAMESPACE::shared::fixunssfti(12.5f));
#endif // LIBC_TYPES_HAS_INT128
}

TEST(LlvmLibcSharedBuiltinsTest, DoubleToIntConversion) {
  EXPECT_EQ(int64_t(12), LIBC_NAMESPACE::shared::fixdfdi(12.5));
  EXPECT_EQ(int32_t(12), LIBC_NAMESPACE::shared::fixdfsi(12.5));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_EQ(static_cast<__int128_t>(12), LIBC_NAMESPACE::shared::fixdfti(12.5));
#endif // LIBC_TYPES_HAS_INT128
  EXPECT_EQ(uint64_t(12), LIBC_NAMESPACE::shared::fixunsdfdi(12.5));
  EXPECT_EQ(uint32_t(12), LIBC_NAMESPACE::shared::fixunsdfsi(12.5));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_EQ(static_cast<__uint128_t>(12),
            LIBC_NAMESPACE::shared::fixunsdfti(12.5));
#endif // LIBC_TYPES_HAS_INT128
}

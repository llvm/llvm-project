//===-- Unittests for shared builtins -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "shared/builtins.h"
#include "src/__support/uint128.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

namespace shared = LIBC_NAMESPACE::shared;

TEST(LlvmLibcSharedBuiltinsTest, AllFloat) {
  EXPECT_FP_EQ(3.0f, shared::addsf3(1.0f, 2.0f));
  EXPECT_FP_EQ(3.0f, shared::divsf3(6.0f, 2.0f));
  EXPECT_FP_EQ(6.0f, shared::mulsf3(2.0f, 3.0f));
  EXPECT_FP_EQ(2.0f, shared::subsf3(5.0f, 3.0f));
}

TEST(LlvmLibcSharedBuiltinsTest, AllDouble) {
  EXPECT_FP_EQ(3.0, shared::adddf3(1.0, 2.0));
  EXPECT_FP_EQ(3.0, shared::divdf3(6.0, 2.0));
  EXPECT_FP_EQ(6.0, shared::muldf3(2.0, 3.0));
  EXPECT_FP_EQ(2.0, shared::subdf3(5.0, 3.0));
}

#ifdef LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedBuiltinsTest, AllFloat128) {
  EXPECT_FP_EQ(float128(3.0), shared::addtf3(float128(1.0), float128(2.0)));
  EXPECT_FP_EQ(float128(3.0), shared::divtf3(float128(6.0), float128(2.0)));
  EXPECT_FP_EQ(float128(6.0), shared::multf3(float128(2.0), float128(3.0)));
  EXPECT_FP_EQ(float128(2.0), shared::subtf3(float128(5.0), float128(3.0)));
}

#endif // LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedBuiltinsTest, IntToDoubleConversion) {
  EXPECT_FP_EQ(12.0, shared::floatdidf(int64_t(12)));
  EXPECT_FP_EQ(12.0, shared::floatsidf(int32_t(12)));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_FP_EQ(12.0, shared::floattidf(static_cast<Int128>(12)));
#endif // LIBC_TYPES_HAS_INT128
}

TEST(LlvmLibcSharedBuiltinsTest, UIntToDoubleConversion) {
  EXPECT_FP_EQ(12.0, shared::floatundidf(uint64_t(12)));
  EXPECT_FP_EQ(12.0, shared::floatunsidf(uint32_t(12)));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_FP_EQ(12.0, shared::floatuntidf(static_cast<UInt128>(12)));
#endif // LIBC_TYPES_HAS_INT128
}

TEST(LlvmLibcSharedBuiltinsTest, IntToFloatConversion) {
  EXPECT_FP_EQ(12.0f, shared::floatdisf(int64_t(12)));
  EXPECT_FP_EQ(12.0f, shared::floatsisf(int32_t(12)));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_FP_EQ(12.0f, shared::floattisf(static_cast<Int128>(12)));
#endif // LIBC_TYPES_HAS_INT128
}

TEST(LlvmLibcSharedBuiltinsTest, UIntToFloatConversion) {
  EXPECT_FP_EQ(12.0f, shared::floatundisf(uint64_t(12)));
  EXPECT_FP_EQ(12.0f, shared::floatunsisf(uint32_t(12)));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_FP_EQ(12.0f, shared::floatuntisf(static_cast<UInt128>(12)));
#endif // LIBC_TYPES_HAS_INT128
}

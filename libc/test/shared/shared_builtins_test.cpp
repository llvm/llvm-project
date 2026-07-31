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

using shared = LIBC_NAMESPACE::shared;

TEST(LlvmLibcSharedBuiltinsTest, SinglePrecisionArithmtic) {
  EXPECT_FP_EQ(3.0f, shared::addsf3(1.0f, 2.0f));
  EXPECT_FP_EQ(3.0f, shared::divsf3(6.0f, 2.0f));
  EXPECT_FP_EQ(6.0f, shared::mulsf3(2.0f, 3.0f));
  EXPECT_FP_EQ(-5.0, shared::negsf2(5.0));
  EXPECT_FP_EQ(2.0f, shared::subsf3(5.0f, 3.0f));
}

TEST(LlvmLibcSharedBuiltinsTest, DoublePrecisionArithmtic) {
  EXPECT_FP_EQ(3.0, shared::adddf3(1.0, 2.0));
  EXPECT_FP_EQ(3.0, shared::divdf3(6.0, 2.0));
  EXPECT_FP_EQ(6.0, shared::muldf3(2.0, 3.0));
  EXPECT_FP_EQ(5.0, shared::negdf2(-5.0));
  EXPECT_FP_EQ(2.0, shared::subdf3(5.0, 3.0));
}

#ifdef LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedBuiltinsTest, QuadPrecisionArithmtic) {
  EXPECT_FP_EQ(float128(3.0), shared::addtf3(float128(1.0), float128(2.0)));
  EXPECT_FP_EQ(float128(3.0), shared::divtf3(float128(6.0), float128(2.0)));
  EXPECT_FP_EQ(float128(6.0), shared::multf3(float128(2.0), float128(3.0)));
  EXPECT_FP_EQ(float128(2.0), shared::subtf3(float128(5.0), float128(3.0)));
}

#endif // LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedBuiltinsTest, ExtendConversion) {
  EXPECT_FP_EQ(1.5, shared::extendsfdf2(1.5f));
  EXPECT_FP_EQ(1.5f, shared::extendbfsf2(static_cast<uint16_t>(0x3FC0)));
#ifdef LIBC_TYPES_HAS_FLOAT128
  EXPECT_FP_EQ(float128(1.5), shared::extenddftf2(1.5));
  EXPECT_FP_EQ(float128(1.5), shared::extendsftf2(1.5f));
#ifdef LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80
  EXPECT_FP_EQ(float128(1.5), shared::extendxftf2(1.5L));
#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80
#endif // LIBC_TYPES_HAS_FLOAT128
}

TEST(LlvmLibcSharedBuiltinsTest, TruncateConversion) {
  EXPECT_EQ(static_cast<uint16_t>(0x3FC0), shared::truncdfbf2(1.5));
  EXPECT_FP_EQ(1.5f, shared::truncdfsf2(1.5));
  EXPECT_EQ(static_cast<uint16_t>(0x3FC0), shared::truncsfbf2(1.5f));
#ifdef LIBC_TYPES_HAS_FLOAT128
  EXPECT_EQ(static_cast<uint16_t>(0x3FC0), shared::trunctfbf2(float128(1.5)));
#endif // LIBC_TYPES_HAS_FLOAT128
#ifdef LIBC_TYPES_HAS_FLOAT128
  EXPECT_FP_EQ(1.5, shared::trunctfdf2(float128(1.5)));
  EXPECT_FP_EQ(1.5f, shared::trunctfsf2(float128(1.5)));
#ifdef LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80
  EXPECT_FP_EQ(1.5L, shared::trunctfxf2(float128(1.5)));
#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80
#endif // LIBC_TYPES_HAS_FLOAT128
}

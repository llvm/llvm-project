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

#ifdef LIBC_TYPES_HAS_NATIVE_FLOAT128

TEST(LlvmLibcSharedBuiltinsTest, QuadPrecisionArithmtic) {
  EXPECT_FP_EQ(float128(3.0), shared::addtf3(float128(1.0), float128(2.0)));
  EXPECT_FP_EQ(float128(3.0), shared::divtf3(float128(6.0), float128(2.0)));
  EXPECT_FP_EQ(float128(6.0), shared::multf3(float128(2.0), float128(3.0)));
  EXPECT_FP_EQ(float128(2.0), shared::subtf3(float128(5.0), float128(3.0)));
}

#endif // LIBC_TYPES_HAS_NATIVE_FLOAT128
#endif // LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedBuiltinsTest, FloatToIntConversion) {
  EXPECT_EQ(int64_t(12), shared::fixsfdi(12.5f));
  EXPECT_EQ(int32_t(12), shared::fixsfsi(12.5f));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_EQ(static_cast<__int128_t>(12), shared::fixsfti(12.5f));
#endif // LIBC_TYPES_HAS_INT128
}

TEST(LlvmLibcSharedBuiltinsTest, FloatToUIntConversion) {
  EXPECT_EQ(uint64_t(12), shared::fixunssfdi(12.5f));
  EXPECT_EQ(uint32_t(12), shared::fixunssfsi(12.5f));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_EQ(static_cast<__uint128_t>(12), shared::fixunssfti(12.5f));
#endif // LIBC_TYPES_HAS_INT128
}

TEST(LlvmLibcSharedBuiltinsTest, DoubleToIntConversion) {
  EXPECT_EQ(int64_t(12), shared::fixdfdi(12.5));
  EXPECT_EQ(int32_t(12), shared::fixdfsi(12.5));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_EQ(static_cast<__int128_t>(12), shared::fixdfti(12.5));
#endif // LIBC_TYPES_HAS_INT128
}

TEST(LlvmLibcSharedBuiltinsTest, DoubleToUIntConversion) {
  EXPECT_EQ(uint64_t(12), shared::fixunsdfdi(12.5));
  EXPECT_EQ(uint32_t(12), shared::fixunsdfsi(12.5));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_EQ(static_cast<__uint128_t>(12), shared::fixunsdfti(12.5));
#endif // LIBC_TYPES_HAS_INT128
}

#ifdef LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedBuiltinsTest, QuadToIntConversion) {
  EXPECT_EQ(int64_t(12), shared::fixtfdi(float128(12.5)));
  EXPECT_EQ(int32_t(12), shared::fixtfsi(float128(12.5)));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_EQ(static_cast<__int128_t>(12), shared::fixtfti(float128(12.5)));
#endif // LIBC_TYPES_HAS_INT128
}

#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedBuiltinsTest, QuadToUIntConversion) {
  EXPECT_EQ(uint64_t(12), shared::fixunstfdi(float128(12.5)));
  EXPECT_EQ(uint32_t(12), shared::fixunstfsi(float128(12.5)));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_EQ(static_cast<__uint128_t>(12), shared::fixunstfti(float128(12.5)));
#endif // LIBC_TYPES_HAS_INT128
}

#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80

TEST(LlvmLibcSharedBuiltinsTest, X86Float80ToIntConversion) {
  EXPECT_EQ(int64_t(12), shared::fixxfdi(12.5L));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_EQ(static_cast<__int128_t>(12), shared::fixxfti(12.5L));
#endif // LIBC_TYPES_HAS_INT128
}

#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80

#ifdef LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80

TEST(LlvmLibcSharedBuiltinsTest, X86Float80ToUIntConversion) {
  EXPECT_EQ(uint64_t(12), shared::fixunsxfdi(12.5L));
  EXPECT_EQ(uint32_t(12), shared::fixunsxfsi(12.5L));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_EQ(static_cast<__uint128_t>(12), shared::fixunsxfti(12.5L));
#endif // LIBC_TYPES_HAS_INT128
}

#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80

TEST(LlvmLibcSharedBuiltinsTest, IntToDoubleConversion) {
  EXPECT_FP_EQ(12.0, shared::floatdidf(int64_t(12)));
  EXPECT_FP_EQ(12.0, shared::floatsidf(int32_t(12)));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_FP_EQ(12.0, shared::floattidf(static_cast<__int128_t>(12)));
#endif // LIBC_TYPES_HAS_INT128
}

TEST(LlvmLibcSharedBuiltinsTest, UIntToDoubleConversion) {
  EXPECT_FP_EQ(12.0, shared::floatundidf(uint64_t(12)));
  EXPECT_FP_EQ(12.0, shared::floatunsidf(uint32_t(12)));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_FP_EQ(12.0, shared::floatuntidf(static_cast<__uint128_t>(12)));
#endif // LIBC_TYPES_HAS_INT128
}

TEST(LlvmLibcSharedBuiltinsTest, IntToFloatConversion) {
  EXPECT_FP_EQ(12.0f, shared::floatdisf(int64_t(12)));
  EXPECT_FP_EQ(12.0f, shared::floatsisf(int32_t(12)));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_FP_EQ(12.0f, shared::floattisf(static_cast<__int128_t>(12)));
#endif // LIBC_TYPES_HAS_INT128
}

TEST(LlvmLibcSharedBuiltinsTest, UIntToFloatConversion) {
  EXPECT_FP_EQ(12.0f, shared::floatundisf(uint64_t(12)));
  EXPECT_FP_EQ(12.0f, shared::floatunsisf(uint32_t(12)));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_FP_EQ(12.0f, shared::floatuntisf(static_cast<__uint128_t>(12)));
#endif // LIBC_TYPES_HAS_INT128
}

#ifdef LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedBuiltinsTest, IntToQuadConversion) {
  EXPECT_FP_EQ(float128(12.0), shared::floatditf(int64_t(12)));
  EXPECT_FP_EQ(float128(12.0), shared::floatsitf(int32_t(12)));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_FP_EQ(float128(12.0), shared::floattitf(static_cast<__int128_t>(12)));
#endif // LIBC_TYPES_HAS_INT128
}

#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedBuiltinsTest, UIntToQuadConversion) {
  EXPECT_FP_EQ(float128(12.0), shared::floatunditf(uint64_t(12)));
  EXPECT_FP_EQ(float128(12.0), shared::floatunsitf(uint32_t(12)));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_FP_EQ(float128(12.0),
               shared::floatuntitf(static_cast<__uint128_t>(12)));
#endif // LIBC_TYPES_HAS_INT128
}

#endif // LIBC_TYPES_HAS_FLOAT128

#ifdef LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80

TEST(LlvmLibcSharedBuiltinsTest, IntToX86Float80Conversion) {
  EXPECT_FP_EQ(12.0L, shared::floatdixf(int64_t(12)));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_FP_EQ(12.0L, shared::floattixf(static_cast<__int128_t>(12)));
#endif // LIBC_TYPES_HAS_INT128
}

#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80

#ifdef LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80

TEST(LlvmLibcSharedBuiltinsTest, UIntToX86Float80Conversion) {
  EXPECT_FP_EQ(12.0L, shared::floatundixf(uint64_t(12)));
#ifdef LIBC_TYPES_HAS_INT128
  EXPECT_FP_EQ(12.0L, shared::floatuntixf(static_cast<__uint128_t>(12)));
#endif // LIBC_TYPES_HAS_INT128
}

#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80

TEST(LlvmLibcSharedBuiltinsTest, ExtendConversion) {
  EXPECT_FP_EQ(1.5, shared::extendsfdf2(1.5f));
#ifdef LIBC_TYPES_HAS_NATIVE_FLOAT128
  EXPECT_FP_EQ(float128(1.5), shared::extenddftf2(1.5));
  EXPECT_FP_EQ(float128(1.5), shared::extendsftf2(1.5f));
#ifdef LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80
  EXPECT_FP_EQ(float128(1.5), shared::extendxftf2(1.5L));
#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80
#endif // LIBC_TYPES_HAS_NATIVE_FLOAT128
}

TEST(LlvmLibcSharedBuiltinsTest, TruncateConversion) {
  EXPECT_FP_EQ(1.5f, shared::truncdfsf2(1.5));
#ifdef LIBC_TYPES_HAS_NATIVE_FLOAT128
  EXPECT_FP_EQ(1.5, shared::trunctfdf2(float128(1.5)));
  EXPECT_FP_EQ(1.5f, shared::trunctfsf2(float128(1.5)));
#ifdef LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80
  EXPECT_FP_EQ(1.5L, shared::trunctfxf2(float128(1.5)));
#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80
#endif // LIBC_TYPES_HAS_NATIVE_FLOAT128
}

TEST(LlvmLibcSharedBuiltinsTest, SingleCompare) {
  const float aNaN =
      LIBC_NAMESPACE::fputil::FPBits<float>::quiet_nan().get_val();
  EXPECT_EQ(-1, shared::gesf2(1.0f, 2.0f));
  EXPECT_EQ(0, shared::gesf2(1.0f, 1.0f));
  EXPECT_EQ(1, shared::gesf2(2.0f, 1.0f));
  EXPECT_EQ(-1, shared::gesf2(aNaN, 1.0f));
  EXPECT_EQ(-1, shared::lesf2(1.0f, 2.0f));
  EXPECT_EQ(0, shared::lesf2(1.0f, 1.0f));
  EXPECT_EQ(1, shared::lesf2(2.0f, 1.0f));
  EXPECT_EQ(1, shared::lesf2(aNaN, 1.0f));
  EXPECT_EQ(0, shared::unordsf2(1.0f, 2.0f));
  EXPECT_EQ(1, shared::unordsf2(aNaN, 1.0f));
}

TEST(LlvmLibcSharedBuiltinsTest, DoubleCompare) {
  const double aNaN =
      LIBC_NAMESPACE::fputil::FPBits<double>::quiet_nan().get_val();
  EXPECT_EQ(-1, shared::gedf2(1.0, 2.0));
  EXPECT_EQ(0, shared::gedf2(1.0, 1.0));
  EXPECT_EQ(1, shared::gedf2(2.0, 1.0));
  EXPECT_EQ(-1, shared::gedf2(aNaN, 1.0));
  EXPECT_EQ(-1, shared::ledf2(1.0, 2.0));
  EXPECT_EQ(0, shared::ledf2(1.0, 1.0));
  EXPECT_EQ(1, shared::ledf2(2.0, 1.0));
  EXPECT_EQ(1, shared::ledf2(aNaN, 1.0));
  EXPECT_EQ(0, shared::unorddf2(1.0, 2.0));
  EXPECT_EQ(1, shared::unorddf2(aNaN, 1.0));
}

#ifdef LIBC_TYPES_HAS_FLOAT128

TEST(LlvmLibcSharedBuiltinsTest, Comparison) {
  const float128 aNaN =
      LIBC_NAMESPACE::fputil::FPBits<float128>::quiet_nan().get_val();
  EXPECT_EQ(-1, shared::getf2(float128(1.0), float128(2.0)));
  EXPECT_EQ(0, shared::getf2(float128(1.0), float128(1.0)));
  EXPECT_EQ(1, shared::getf2(float128(2.0), float128(1.0)));
  EXPECT_EQ(-1, shared::getf2(aNaN, float128(1.0)));
  EXPECT_EQ(-1, shared::letf2(float128(1.0), float128(2.0)));
  EXPECT_EQ(0, shared::letf2(float128(1.0), float128(1.0)));
  EXPECT_EQ(1, shared::letf2(float128(2.0), float128(1.0)));
  EXPECT_EQ(1, shared::letf2(aNaN, float128(1.0)));
  EXPECT_EQ(0, shared::unordtf2(float128(1.0), float128(2.0)));
  EXPECT_EQ(1, shared::unordtf2(aNaN, float128(1.0)));
}

#endif // LIBC_TYPES_HAS_FLOAT128

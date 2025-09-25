//===-- Unittests for complex ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/complex-macros.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcComplexTest, VersionMacro) {
  EXPECT_EQ(__STDC_VERSION_COMPLEX_H__, 202311L);
}

TEST(LlvmLibcComplexTest, IMacro) { EXPECT_CFP_EQ(1.0fi, I); }

TEST(LlvmLibcComplexTest, _Complex_IMacro) { EXPECT_CFP_EQ(1.0fi, _Complex_I); }

TEST(LlvmLibcComplexTest, CMPLXMacro) {
  EXPECT_CFP_EQ(CMPLX(0, 1.0), I);
  EXPECT_CFP_EQ(CMPLX(1.0, 0), 1.0);
  EXPECT_CFP_EQ(CMPLXF(0, 1.0f), I);
  EXPECT_CFP_EQ(CMPLXF(1.0f, 0), 1.0f);
  EXPECT_CFP_EQ(CMPLXL(0, 1.0l), I);
  EXPECT_CFP_EQ(CMPLXL(1.0l, 0), 1.0l);

#ifdef LIBC_TYPES_HAS_CFLOAT16
  EXPECT_CFP_EQ(CMPLXF16(0, 1.0), I);
  EXPECT_CFP_EQ(CMPLXF16(1.0, 0), 1.0);
#endif // LIBC_TYPES_HAS_CFLOAT16

#ifdef LIBC_TYPES_HAS_CFLOAT128
  EXPECT_CFP_EQ(CMPLXF128(0, 1.0), I);
  EXPECT_CFP_EQ(CMPLXF128(1.0, 0), 1.0);
#endif // LIBC_TYPES_HAS_CFLOAT128
}

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

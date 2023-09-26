//===-- Unittests for isalpha----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isalpha.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcIsAlpha, DefaultLocale) {
  // Loops through all characters, verifying that letters return a
  // non-zero integer and everything else returns zero.
  for (int ch = -255; ch < 255; ++ch) {
    if (('a' <= ch && ch <= 'z') || ('A' <= ch && ch <= 'Z'))
      EXPECT_NE(LIBC_NAMESPACE::isalpha(ch), 0);
    else
      EXPECT_EQ(LIBC_NAMESPACE::isalpha(ch), 0);
  }
}

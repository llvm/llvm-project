//===-- Unittests for isalnum----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isalnum.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcIsAlNum, DefaultLocale) {
  // Loops through all characters, verifying that numbers and letters
  // return non-zero integer and everything else returns a zero.
  for (int c = -255; c < 255; ++c) {
    if (('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') ||
        ('0' <= c && c <= '9'))
      EXPECT_NE(LIBC_NAMESPACE::isalnum(c), 0);
    else
      EXPECT_EQ(LIBC_NAMESPACE::isalnum(c), 0);
  }
}

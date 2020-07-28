//===-- Unittests for isalpha----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isalpha.h"
#include "utils/UnitTest/Test.h"

// Helper function that makes a call to isalpha a bit cleaner
// for use with testing utilities, since it explicitly requires
// a boolean value for EXPECT_TRUE and EXPECT_FALSE.
bool call_isalpha(int c) { return __llvm_libc::isalpha(c); }

TEST(IsAlpha, DefaultLocale) {
  // Loops through all characters, verifying that letters return true
  // and everything else returns false.
  for (int ch = 0; ch < 255; ++ch) {
    if (('a' <= ch && ch <= 'z') || ('A' <= ch && ch <= 'Z'))
      EXPECT_TRUE(call_isalpha(ch));
    else
      EXPECT_FALSE(call_isalpha(ch));
  }
}

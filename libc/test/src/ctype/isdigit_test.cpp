//===-- Unittests for isdigit----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isdigit.h"
#include "utils/UnitTest/Test.h"

// Helper function that makes a call to isdigit a bit cleaner
// for use with testing utilities, since it explicitly requires
// a boolean value for EXPECT_TRUE and EXPECT_FALSE.
bool call_isdigit(int c) { return __llvm_libc::isdigit(c); }

TEST(IsDigit, DefaultLocale) {
  // Loops through all characters, verifying that numbers return true
  // and everything else returns false.
  for (int ch = 0; ch < 255; ++ch) {
    if ('0' <= ch && ch <= '9')
      EXPECT_TRUE(call_isdigit(ch));
    else
      EXPECT_FALSE(call_isdigit(ch));
  }
}

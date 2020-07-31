//===-- Unittests for isalnum----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isalnum.h"
#include "utils/UnitTest/Test.h"

// Helper function that makes a call to isalnum a bit cleaner
// for use with testing utilities, since it explicitly requires
// a boolean value for EXPECT_TRUE and EXPECT_FALSE.
bool call_isalnum(int c) { return __llvm_libc::isalnum(c); }

TEST(IsAlNum, DefaultLocale) {
  // Loops through all characters, verifying that numbers and letters
  // return true and everything else returns false.
  for (int c = 0; c < 255; ++c) {
    if (('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') ||
        ('0' <= c && c <= '9'))
      EXPECT_TRUE(call_isalnum(c));
    else
      EXPECT_FALSE(call_isalnum(c));
  }
}

//===-- Unittests for isspace----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isspace.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcIsSpace, DefaultLocale) {
  // Loops through all characters, verifying that space characters
  // return true and everything else returns false.
  // Hexadecimal | Symbol
  // ---------------------------
  //    0x09     |   horizontal tab
  //    0x0a     |   line feed
  //    0x0b     |   vertical tab
  //    0x0d     |   carriage return
  //    0x20     |   space
  for (int ch = -255; ch < 255; ++ch) {
    if (ch == 0x20 || (0x09 <= ch && ch <= 0x0d))
      EXPECT_NE(__llvm_libc::isspace(ch), 0);
    else
      EXPECT_EQ(__llvm_libc::isspace(ch), 0);
  }
}

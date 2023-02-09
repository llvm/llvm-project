//===-- Unittests for toascii----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/toascii.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcToAscii, DefaultLocale) {
  // Loops through all characters, verifying that ascii characters
  //    (which are all 7 bit unsigned integers)
  // return themself, and that all other characters return themself
  // mod 128 (which is equivalent to & 0x7f)
  for (int ch = 0; ch < 255; ++ch) {
    if (ch <= 0x7f)
      EXPECT_EQ(__llvm_libc::toascii(ch), ch);
    else
      EXPECT_EQ(__llvm_libc::toascii(ch), ch & 0x7f);
  }
}

//===-- Unittests for isblank----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isblank.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcIsBlank, DefaultLocale) {
  // Loops through all characters, verifying that space and horizontal tab
  // return a non-zero integer and everything else returns zero.
  for (int ch = 0; ch < 255; ++ch) {
    if (ch == ' ' || ch == '\t')
      EXPECT_NE(__llvm_libc::isblank(ch), 0);
    else
      EXPECT_EQ(__llvm_libc::isblank(ch), 0);
  }
}

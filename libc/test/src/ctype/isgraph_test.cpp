//===-- Unittests for isgraph----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isgraph.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcIsGraph, DefaultLocale) {
  // Loops through all characters, verifying that graphical characters
  // return a non-zero integer, everything else returns zero.
  for (int ch = 0; ch < 255; ++ch) {
    if ('!' <= ch && ch <= '~') // A-Z, a-z, 0-9, punctuation.
      EXPECT_NE(__llvm_libc::isgraph(ch), 0);
    else
      EXPECT_EQ(__llvm_libc::isgraph(ch), 0);
  }
}

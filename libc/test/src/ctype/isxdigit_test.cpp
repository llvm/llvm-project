//===-- Unittests for isxdigit---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isxdigit.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcIsXDigit, DefaultLocale) {
  for (int ch = -255; ch < 255; ++ch) {
    if (('0' <= ch && ch <= '9') || ('a' <= ch && ch <= 'f') ||
        ('A' <= ch && ch <= 'F'))
      EXPECT_NE(__llvm_libc::isxdigit(ch), 0);
    else
      EXPECT_EQ(__llvm_libc::isxdigit(ch), 0);
  }
}

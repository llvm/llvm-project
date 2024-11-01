//===-- Unittests for wctob ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h> //for EOF

#include "src/wchar/wctob.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcWctob, DefaultLocale) {
  // Loops through a subset of the wide characters, verifying that ascii returns
  // itself and everything else returns EOF.
  for (wint_t c = 0; c < 32767; ++c) {
    if (c < 128)
      EXPECT_EQ(LIBC_NAMESPACE::wctob(c), static_cast<int>(c));
    else
      EXPECT_EQ(LIBC_NAMESPACE::wctob(c), EOF);
  }
}

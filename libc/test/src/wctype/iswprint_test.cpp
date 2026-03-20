//===-- Unittests for iswprint --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/iswprint.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibciswprint, SimpleTest) {
  for (int ch = -255; ch < 255; ++ch) {
    if (' ' <= ch && ch <= '~') {
      EXPECT_NE(LIBC_NAMESPACE::iswprint(ch), 0);
    } else {
      EXPECT_EQ(LIBC_NAMESPACE::iswprint(ch), 0);
    }
  }
}

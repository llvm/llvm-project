//===-- Unittests for btowc ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <wchar.h> //for WEOF

#include "src/wchar/btowc.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcBtowc, DefaultLocale) {
  // Loops through all characters, verifying that ascii returns itself and
  // everything else returns WEOF.
  for (int c = 0; c < 255; ++c) {
    if (c < 128)
      EXPECT_EQ(LIBC_NAMESPACE::btowc(c), static_cast<wint_t>(c));
    else
      EXPECT_EQ(LIBC_NAMESPACE::btowc(c), WEOF);
  }
}

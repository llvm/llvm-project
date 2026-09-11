//===-- Unittests for iswprint --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/wctype_utils.h"
#include "src/wctype/iswprint.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibciswprint, SimpleTest) {
  for (int ch = -255; ch < 255; ++ch) {
    if (' ' <= ch && ch <= '~') {
      EXPECT_NE(LIBC_NAMESPACE::iswprint(ch), 0);
    } else {
#if LIBC_CONF_WCTYPE_MODE == LIBC_WCTYPE_MODE_UTF8
      // In UTF-8 mode, characters above 127 can be printable.
      if (ch < 128) {
        EXPECT_EQ(LIBC_NAMESPACE::iswprint(ch), 0);
      }
#else
      // In ASCII mode, everything outside ASCII printable is not printable.
      EXPECT_EQ(LIBC_NAMESPACE::iswprint(ch), 0);
#endif
    }
  }

#if LIBC_CONF_WCTYPE_MODE == LIBC_WCTYPE_MODE_UTF8
  EXPECT_NE(LIBC_NAMESPACE::iswprint(L'á'), 0);
#else
  EXPECT_EQ(LIBC_NAMESPACE::iswprint(L'á'), 0);
#endif
}

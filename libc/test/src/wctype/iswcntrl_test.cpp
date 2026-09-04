//===-- Unittests for iswcntrl --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/iswcntrl.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibciswcntrl, DefaultLocale) {
  EXPECT_NE(LIBC_NAMESPACE::iswcntrl(L'\0'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswcntrl(L'\t'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswcntrl(L'\n'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswcntrl(L'\v'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswcntrl(L'\f'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswcntrl(L'\r'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswcntrl(L' '), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswcntrl(L'!'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswcntrl(L'0'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswcntrl(L'9'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswcntrl(L'A'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswcntrl(L'Z'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswcntrl(L'a'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswcntrl(L'z'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswcntrl(L'?'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswcntrl(L'~'), 0);
}

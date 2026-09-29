//===-- Unittests for strcasestr ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcasestr.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrCaseStrTest, NeedleNotInHaystack) {
  EXPECT_STREQ(LIBC_NAMESPACE::strcasestr("abcd", "e"), nullptr);
  EXPECT_STREQ(LIBC_NAMESPACE::strcasestr("ABCD", "e"), nullptr);
  EXPECT_STREQ(LIBC_NAMESPACE::strcasestr("abcd", "E"), nullptr);
  EXPECT_STREQ(LIBC_NAMESPACE::strcasestr("ABCD", "E"), nullptr);
}

TEST(LlvmLibcStrCaseStrTest, NeedleInMiddle) {
  EXPECT_STREQ(LIBC_NAMESPACE::strcasestr("abcdefghi", "def"), "defghi");
  EXPECT_STREQ(LIBC_NAMESPACE::strcasestr("ABCDEFGHI", "def"), "DEFGHI");
  EXPECT_STREQ(LIBC_NAMESPACE::strcasestr("abcdefghi", "DEF"), "defghi");
  EXPECT_STREQ(LIBC_NAMESPACE::strcasestr("ABCDEFGHI", "DEF"), "DEFGHI");
}

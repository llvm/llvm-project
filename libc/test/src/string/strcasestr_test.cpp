//===-- Unittests for strcasestr ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcasestr.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcStrCaseStrTest, NeedleNotInHaystack) {
  EXPECT_STREQ(__llvm_libc::strcasestr("abcd", "e"), nullptr);
  EXPECT_STREQ(__llvm_libc::strcasestr("ABCD", "e"), nullptr);
  EXPECT_STREQ(__llvm_libc::strcasestr("abcd", "E"), nullptr);
  EXPECT_STREQ(__llvm_libc::strcasestr("ABCD", "E"), nullptr);
}

TEST(LlvmLibcStrCaseStrTest, NeedleInMiddle) {
  EXPECT_STREQ(__llvm_libc::strcasestr("abcdefghi", "def"), "defghi");
  EXPECT_STREQ(__llvm_libc::strcasestr("ABCDEFGHI", "def"), "DEFGHI");
  EXPECT_STREQ(__llvm_libc::strcasestr("abcdefghi", "DEF"), "defghi");
  EXPECT_STREQ(__llvm_libc::strcasestr("ABCDEFGHI", "DEF"), "DEFGHI");
}

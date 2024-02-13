//===-- Unittests for strsep ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strsep.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrsepTest, NullSrc) {
  char *string = nullptr;
  EXPECT_STREQ(LIBC_NAMESPACE::strsep(&string, ""), nullptr);
}

TEST(LlvmLibcStrsepTest, NoTokenFound) {
  {
    char s[] = "";
    char *string = s, *orig = s;
    EXPECT_STREQ(LIBC_NAMESPACE::strsep(&string, ""), nullptr);
    EXPECT_EQ(orig, string);
  }
  {
    char s[] = "abcde";
    char *string = s, *orig = s;
    EXPECT_STREQ(LIBC_NAMESPACE::strsep(&string, ""), orig);
    EXPECT_EQ(string, orig + 5);
  }
  {
    char s[] = "abcde";
    char *string = s, *orig = s;
    EXPECT_STREQ(LIBC_NAMESPACE::strsep(&string, "fghijk"), orig);
    EXPECT_EQ(string, orig + 5);
  }
}

TEST(LlvmLibcStrsepTest, TokenFound) {
  char s[] = "abacd";
  char *string = s;
  EXPECT_STREQ(LIBC_NAMESPACE::strsep(&string, "c"), "aba");
  EXPECT_STREQ(string, "d");
}

TEST(LlvmLibcStrsepTest, DelimitersShouldNotBeIncludedInToken) {
  char s[] = "__ab__:cd_:_ef_:_";
  char *string = s;
  const char *expected[] = {"", "",   "ab", "", "", "cd",   "",
                            "", "ef", "",   "", "", nullptr};
  for (int i = 0; expected[i]; i++) {
    ASSERT_STREQ(LIBC_NAMESPACE::strsep(&string, "_:"), expected[i]);
  }
}

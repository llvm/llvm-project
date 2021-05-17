//===-- Unittests for strtok ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strtok.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcStrTokTest, NoTokenFound) {
  char empty[] = "";
  ASSERT_STREQ(__llvm_libc::strtok(empty, ""), nullptr);
  ASSERT_STREQ(__llvm_libc::strtok(empty, "_"), nullptr);

  char single[] = "_";
  ASSERT_STREQ(__llvm_libc::strtok(single, ""), "_");

  char multiple[] = "1,2";
  ASSERT_STREQ(__llvm_libc::strtok(multiple, ":"), "1,2");
}

TEST(LlvmLibcStrTokTest, DelimiterAsFirstCharacterShouldBeIgnored) {
  char src[] = ".123";
  ASSERT_STREQ(__llvm_libc::strtok(src, "."), "123");
}

TEST(LlvmLibcStrTokTest, DelimiterIsMiddleCharacter) {
  char src[] = "12,34";
  ASSERT_STREQ(__llvm_libc::strtok(src, ","), "12");
}

TEST(LlvmLibcStrTokTest, DelimiterAsLastCharacterShouldBeIgnored) {
  char src[] = "1234:";
  ASSERT_STREQ(__llvm_libc::strtok(src, ":"), "1234");
}

TEST(LlvmLibcStrTokTest, MultipleDelimiters) {
  char src[] = "12,.34";
  ASSERT_STREQ(__llvm_libc::strtok(src, "."), "12,");
  ASSERT_STREQ(__llvm_libc::strtok(src, ".,"), "12");
  ASSERT_STREQ(__llvm_libc::strtok(src, ",."), "12");
  ASSERT_STREQ(__llvm_libc::strtok(src, ":,."), "12");
}

TEST(LlvmLibcStrTokTest, ShouldNotGoPastNullTerminator) {
  char src[] = {'1', '2', '\0', ',', '3'};
  ASSERT_STREQ(__llvm_libc::strtok(src, ","), "12");
}

TEST(LlvmLibcStrTokTest, SubsequentCallsShouldFindFollowingDelimiters) {
  char src[] = "12,34.56";
  char *token = __llvm_libc::strtok(src, ",.");
  ASSERT_STREQ(token, "12");
  token = __llvm_libc::strtok(nullptr, ",.");
  ASSERT_STREQ(token, "34");
  token = __llvm_libc::strtok(nullptr, ",.");
  ASSERT_STREQ(token, "56");
  token = __llvm_libc::strtok(nullptr, "_:,_");
  ASSERT_STREQ(token, nullptr);
  // Subsequent calls after hitting the end of the string should also return
  // nullptr.
  token = __llvm_libc::strtok(nullptr, "_:,_");
  ASSERT_STREQ(token, nullptr);
}

TEST(LlvmLibcStrTokTest, DelimitersShouldNotBeIncludedInToken) {
  char src[] = "__ab__:_cd__:__ef__:__";
  char *token = __llvm_libc::strtok(src, "_:");
  ASSERT_STREQ(token, "ab");
  token = __llvm_libc::strtok(nullptr, ":_");
  ASSERT_STREQ(token, "cd");
  token = __llvm_libc::strtok(nullptr, "_:,");
  ASSERT_STREQ(token, "ef");
  token = __llvm_libc::strtok(nullptr, "_:,_");
  ASSERT_STREQ(token, nullptr);
}

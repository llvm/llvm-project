//===-- Unittests for strtok_r -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strtok_r.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrTokReentrantTest, NoTokenFound) {
  { // Empty source and delimiter string.
    char empty[] = "";
    char *reserve = nullptr;
    ASSERT_STREQ(__llvm_libc::strtok_r(empty, "", &reserve), nullptr);
    // Another call to ensure that 'reserve' is not in a bad state.
    ASSERT_STREQ(__llvm_libc::strtok_r(empty, "", &reserve), nullptr);
    ASSERT_STREQ(__llvm_libc::strtok_r(nullptr, "", &reserve), nullptr);
  }
  { // Empty source and single character delimiter string.
    char empty[] = "";
    char *reserve = nullptr;
    ASSERT_STREQ(__llvm_libc::strtok_r(empty, "_", &reserve), nullptr);
    // Another call to ensure that 'reserve' is not in a bad state.
    ASSERT_STREQ(__llvm_libc::strtok_r(empty, "_", &reserve), nullptr);
    ASSERT_STREQ(__llvm_libc::strtok_r(nullptr, "_", &reserve), nullptr);
  }
  { // Same character source and delimiter string.
    char single[] = "_";
    char *reserve = nullptr;
    ASSERT_STREQ(__llvm_libc::strtok_r(single, "_", &reserve), nullptr);
    // Another call to ensure that 'reserve' is not in a bad state.
    ASSERT_STREQ(__llvm_libc::strtok_r(single, "_", &reserve), nullptr);
    ASSERT_STREQ(__llvm_libc::strtok_r(nullptr, "_", &reserve), nullptr);
  }
  { // Multiple character source and single character delimiter string.
    char multiple[] = "1,2";
    char *reserve = nullptr;
    ASSERT_STREQ(__llvm_libc::strtok_r(multiple, ":", &reserve), "1,2");
    // Another call to ensure that 'reserve' is not in a bad state.
    ASSERT_STREQ(__llvm_libc::strtok_r(multiple, ":", &reserve), "1,2");
    ASSERT_STREQ(__llvm_libc::strtok_r(nullptr, ":", &reserve), nullptr);
  }
}

TEST(LlvmLibcStrTokReentrantTest, DelimiterAsFirstCharacterShouldBeIgnored) {
  char src[] = ".123";
  char *reserve = nullptr;
  ASSERT_STREQ(__llvm_libc::strtok_r(src, ".", &reserve), "123");
  // Another call to ensure that 'reserve' is not in a bad state.
  ASSERT_STREQ(__llvm_libc::strtok_r(src, ".", &reserve), "123");
  ASSERT_STREQ(__llvm_libc::strtok_r(nullptr, ".", &reserve), nullptr);
}

TEST(LlvmLibcStrTokReentrantTest, DelimiterIsMiddleCharacter) {
  char src[] = "12,34";
  char *reserve = nullptr;
  ASSERT_STREQ(__llvm_libc::strtok_r(src, ",", &reserve), "12");
  // Another call to ensure that 'reserve' is not in a bad state.
  ASSERT_STREQ(__llvm_libc::strtok_r(src, ",", &reserve), "12");
  ASSERT_STREQ(__llvm_libc::strtok_r(nullptr, ",", &reserve), nullptr);
}

TEST(LlvmLibcStrTokReentrantTest, DelimiterAsLastCharacterShouldBeIgnored) {
  char src[] = "1234:";
  char *reserve = nullptr;
  ASSERT_STREQ(__llvm_libc::strtok_r(src, ":", &reserve), "1234");
  // Another call to ensure that 'reserve' is not in a bad state.
  ASSERT_STREQ(__llvm_libc::strtok_r(src, ":", &reserve), "1234");
  ASSERT_STREQ(__llvm_libc::strtok_r(nullptr, ":", &reserve), nullptr);
}

TEST(LlvmLibcStrTokReentrantTest, ShouldNotGoPastNullTerminator) {
  char src[] = {'1', '2', '\0', ',', '3'};
  char *reserve = nullptr;
  ASSERT_STREQ(__llvm_libc::strtok_r(src, ",", &reserve), "12");
  // Another call to ensure that 'reserve' is not in a bad state.
  ASSERT_STREQ(__llvm_libc::strtok_r(src, ",", &reserve), "12");
  ASSERT_STREQ(__llvm_libc::strtok_r(nullptr, ",", &reserve), nullptr);
}

TEST(LlvmLibcStrTokReentrantTest,
     ShouldReturnNullptrWhenBothSrcAndSaveptrAreNull) {
  char *src = nullptr;
  char *reserve = nullptr;
  // Ensure that instead of crashing if src and reserve are null, nullptr is
  // returned
  ASSERT_STREQ(__llvm_libc::strtok_r(src, ",", &reserve), nullptr);
  // And that neither src nor reserve are changed when that happens
  ASSERT_STREQ(src, nullptr);
  ASSERT_STREQ(reserve, nullptr);
}

TEST(LlvmLibcStrTokReentrantTest,
     SubsequentCallsShouldFindFollowingDelimiters) {
  char src[] = "12,34.56";
  char *reserve = nullptr;
  char *token = __llvm_libc::strtok_r(src, ",.", &reserve);
  ASSERT_STREQ(token, "12");
  token = __llvm_libc::strtok_r(nullptr, ",.", &reserve);
  ASSERT_STREQ(token, "34");
  token = __llvm_libc::strtok_r(nullptr, ",.", &reserve);
  ASSERT_STREQ(token, "56");
  token = __llvm_libc::strtok_r(nullptr, "_:,_", &reserve);
  ASSERT_STREQ(token, nullptr);
  // Subsequent calls after hitting the end of the string should also return
  // nullptr.
  token = __llvm_libc::strtok_r(nullptr, "_:,_", &reserve);
  ASSERT_STREQ(token, nullptr);
}

TEST(LlvmLibcStrTokReentrantTest, DelimitersShouldNotBeIncludedInToken) {
  char src[] = "__ab__:_cd__:__ef__:__";
  char *reserve = nullptr;
  char *token = __llvm_libc::strtok_r(src, "_:", &reserve);
  ASSERT_STREQ(token, "ab");
  token = __llvm_libc::strtok_r(nullptr, ":_", &reserve);
  ASSERT_STREQ(token, "cd");
  token = __llvm_libc::strtok_r(nullptr, "_:,", &reserve);
  ASSERT_STREQ(token, "ef");
  token = __llvm_libc::strtok_r(nullptr, "_:,_", &reserve);
  ASSERT_STREQ(token, nullptr);
}

//===-- Unittests for strchr ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strchr.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcStrChrTest, FindsFirstCharacter) {
  const char *src = "abcde";

  // Should return original string since 'a' is the first character.
  ASSERT_STREQ(__llvm_libc::strchr(src, 'a'), "abcde");
  // Source string should not change.
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrChrTest, FindsMiddleCharacter) {
  const char *src = "abcde";

  // Should return characters after (and including) 'c'.
  ASSERT_STREQ(__llvm_libc::strchr(src, 'c'), "cde");
  // Source string should not change.
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrChrTest, FindsLastCharacterThatIsNotNullTerminator) {
  const char *src = "abcde";

  // Should return 'e' and null-terminator.
  ASSERT_STREQ(__llvm_libc::strchr(src, 'e'), "e");
  // Source string should not change.
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrChrTest, FindsNullTerminator) {
  const char *src = "abcde";

  // Should return null terminator.
  ASSERT_STREQ(__llvm_libc::strchr(src, '\0'), "");
  // Source string should not change.
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrChrTest, CharacterNotWithinStringShouldReturnNullptr) {
  // Since 'z' is not within the string, should return nullptr.
  ASSERT_STREQ(__llvm_libc::strchr("123?", 'z'), nullptr);
}

TEST(LlvmLibcStrChrTest, TheSourceShouldNotChange) {
  const char *src = "abcde";
  // When the character is found, the source string should not change.
  __llvm_libc::strchr(src, 'd');
  ASSERT_STREQ(src, "abcde");
  // Same case for when the character is not found.
  __llvm_libc::strchr(src, 'z');
  ASSERT_STREQ(src, "abcde");
  // Same case for when looking for nullptr.
  __llvm_libc::strchr(src, '\0');
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrChrTest, ShouldFindFirstOfDuplicates) {
  // '1' is duplicated in the string, but it should find the first copy.
  ASSERT_STREQ(__llvm_libc::strchr("abc1def1ghi", '1'), "1def1ghi");

  const char *dups = "XXXXX";
  // Should return original string since 'X' is the first character.
  ASSERT_STREQ(__llvm_libc::strchr(dups, 'X'), dups);
}

TEST(LlvmLibcStrChrTest, EmptyStringShouldOnlyMatchNullTerminator) {
  // Null terminator should match.
  ASSERT_STREQ(__llvm_libc::strchr("", '\0'), "");
  // All other characters should not match.
  ASSERT_STREQ(__llvm_libc::strchr("", 'Z'), nullptr);
  ASSERT_STREQ(__llvm_libc::strchr("", '3'), nullptr);
  ASSERT_STREQ(__llvm_libc::strchr("", '*'), nullptr);
}

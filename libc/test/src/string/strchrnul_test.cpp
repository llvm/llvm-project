//===-- Unittests for strchrnul -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strchrnul.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrChrNulTest, FindsFirstCharacter) {
  const char *src = "abcde";

  // Should return original string since 'a' is the first character.
  ASSERT_STREQ(__llvm_libc::strchrnul(src, 'a'), "abcde");
  // Source string should not change.
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrChrNulTest, FindsMiddleCharacter) {
  const char *src = "abcde";

  // Should return characters after (and including) 'c'.
  ASSERT_STREQ(__llvm_libc::strchrnul(src, 'c'), "cde");
  // Source string should not change.
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrChrNulTest, FindsLastCharacterThatIsNotNullTerminator) {
  const char *src = "abcde";

  // Should return 'e' and null-terminator.
  ASSERT_STREQ(__llvm_libc::strchrnul(src, 'e'), "e");
  // Source string should not change.
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrChrNulTest, FindsNullTerminator) {
  const char *src = "abcde";

  // Should return null terminator.
  ASSERT_STREQ(__llvm_libc::strchrnul(src, '\0'), "");
  // Source string should not change.
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrChrNulTest,
     CharacterNotWithinStringShouldReturnNullTerminator) {
  const char *src = "123?";

  // Since 'z' is not within the string, should return a pointer to the source
  // string's null terminator.
  char *result = __llvm_libc::strchrnul(src, 'z');
  ASSERT_EQ(*result, '\0');

  char *term = const_cast<char *>(src) + 4;
  ASSERT_EQ(result, term);
}

TEST(LlvmLibcStrChrNulTest, TheSourceShouldNotChange) {
  const char *src = "abcde";
  // When the character is found, the source string should not change.
  __llvm_libc::strchrnul(src, 'd');
  ASSERT_STREQ(src, "abcde");
  // Same case for when the character is not found.
  __llvm_libc::strchrnul(src, 'z');
  ASSERT_STREQ(src, "abcde");
  // Same case for when looking for null terminator.
  __llvm_libc::strchrnul(src, '\0');
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrChrNulTest, ShouldFindFirstOfDuplicates) {
  // '1' is duplicated in the string, but it should find the first copy.
  ASSERT_STREQ(__llvm_libc::strchrnul("abc1def1ghi", '1'), "1def1ghi");

  const char *dups = "XXXXX";
  // Should return original string since 'X' is the first character.
  ASSERT_STREQ(__llvm_libc::strchrnul(dups, 'X'), dups);
}

TEST(LlvmLibcStrChrNulTest, EmptyStringShouldOnlyMatchNullTerminator) {
  // Null terminator should match.
  ASSERT_STREQ(__llvm_libc::strchrnul("", '\0'), "");

  // All other characters should not match.
  char *result = __llvm_libc::strchrnul("", 'Z');
  ASSERT_EQ(*result, '\0');

  result = __llvm_libc::strchrnul("", '3');
  ASSERT_EQ(*result, '\0');

  result = __llvm_libc::strchrnul("", '*');
  ASSERT_EQ(*result, '\0');
}

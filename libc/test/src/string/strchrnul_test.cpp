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
  ASSERT_STREQ(LIBC_NAMESPACE::strchrnul(src, 'a'), "abcde");
  // Source string should not change.
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrChrNulTest, FindsMiddleCharacter) {
  const char *src = "abcde";

  // Should return characters after (and including) 'c'.
  ASSERT_STREQ(LIBC_NAMESPACE::strchrnul(src, 'c'), "cde");
  // Source string should not change.
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrChrNulTest, FindsLastCharacterThatIsNotNullTerminator) {
  const char *src = "abcde";

  // Should return 'e' and null-terminator.
  ASSERT_STREQ(LIBC_NAMESPACE::strchrnul(src, 'e'), "e");
  // Source string should not change.
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrChrNulTest, FindsNullTerminator) {
  const char *src = "abcde";

  // Should return null terminator.
  ASSERT_STREQ(LIBC_NAMESPACE::strchrnul(src, '\0'), "");
  // Source string should not change.
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrChrNulTest,
     CharacterNotWithinStringShouldReturnNullTerminator) {
  const char *src = "123?";

  // Since 'z' is not within the string, should return a pointer to the source
  // string's null terminator.
  char *result = LIBC_NAMESPACE::strchrnul(src, 'z');
  ASSERT_EQ(*result, '\0');

  char *term = const_cast<char *>(src) + 4;
  ASSERT_EQ(result, term);
}

TEST(LlvmLibcStrChrNulTest, TheSourceShouldNotChange) {
  const char *src = "abcde";
  // When the character is found, the source string should not change.
  LIBC_NAMESPACE::strchrnul(src, 'd');
  ASSERT_STREQ(src, "abcde");
  // Same case for when the character is not found.
  LIBC_NAMESPACE::strchrnul(src, 'z');
  ASSERT_STREQ(src, "abcde");
  // Same case for when looking for null terminator.
  LIBC_NAMESPACE::strchrnul(src, '\0');
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrChrNulTest, ShouldFindFirstOfDuplicates) {
  // '1' is duplicated in the string, but it should find the first copy.
  ASSERT_STREQ(LIBC_NAMESPACE::strchrnul("abc1def1ghi", '1'), "1def1ghi");

  const char *dups = "XXXXX";
  // Should return original string since 'X' is the first character.
  ASSERT_STREQ(LIBC_NAMESPACE::strchrnul(dups, 'X'), dups);
}

TEST(LlvmLibcStrChrNulTest, EmptyStringShouldOnlyMatchNullTerminator) {
  // Null terminator should match.
  ASSERT_STREQ(LIBC_NAMESPACE::strchrnul("", '\0'), "");

  // All other characters should not match.
  char *result = LIBC_NAMESPACE::strchrnul("", 'Z');
  ASSERT_EQ(*result, '\0');

  result = LIBC_NAMESPACE::strchrnul("", '3');
  ASSERT_EQ(*result, '\0');

  result = LIBC_NAMESPACE::strchrnul("", '*');
  ASSERT_EQ(*result, '\0');
}

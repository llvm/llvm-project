//===-- Unittests for strrchr ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strrchr.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrRChrTest, FindsFirstCharacter) {
  const char *src = "abcde";

  // Should return original string since 'a' is the first character.
  ASSERT_STREQ(__llvm_libc::strrchr(src, 'a'), "abcde");
  // Source string should not change.
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrRChrTest, FindsMiddleCharacter) {
  const char *src = "abcde";

  // Should return characters after (and including) 'c'.
  ASSERT_STREQ(__llvm_libc::strrchr(src, 'c'), "cde");
  // Source string should not change.
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrRChrTest, FindsLastCharacterThatIsNotNullTerminator) {
  const char *src = "abcde";

  // Should return 'e' and null-terminator.
  ASSERT_STREQ(__llvm_libc::strrchr(src, 'e'), "e");
  // Source string should not change.
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrRChrTest, FindsNullTerminator) {
  const char *src = "abcde";

  // Should return null terminator.
  ASSERT_STREQ(__llvm_libc::strrchr(src, '\0'), "");
  // Source string should not change.
  ASSERT_STREQ(src, "abcde");
}

TEST(LlvmLibcStrRChrTest, FindsLastBehindFirstNullTerminator) {
  const char src[6] = {'a', 'a', '\0', 'b', '\0', 'c'};
  // 'b' is behind a null terminator, so should not be found.
  ASSERT_STREQ(__llvm_libc::strrchr(src, 'b'), nullptr);
  // Same goes for 'c'.
  ASSERT_STREQ(__llvm_libc::strrchr(src, 'c'), nullptr);
  
  // Should find the second of the two a's.
  ASSERT_STREQ(__llvm_libc::strrchr(src, 'a'), "a");
}

TEST(LlvmLibcStrRChrTest, CharacterNotWithinStringShouldReturnNullptr) {
  // Since 'z' is not within the string, should return nullptr.
  ASSERT_STREQ(__llvm_libc::strrchr("123?", 'z'), nullptr);
}

TEST(LlvmLibcStrRChrTest, ShouldFindLastOfDuplicates) {
  // '1' is duplicated in the string, but it should find the last copy.
  ASSERT_STREQ(__llvm_libc::strrchr("abc1def1ghi", '1'), "1ghi");

  const char *dups = "XXXXX";
  // Should return the last occurrence of 'X'.
  ASSERT_STREQ(__llvm_libc::strrchr(dups, 'X'), "X");
}

TEST(LlvmLibcStrRChrTest, EmptyStringShouldOnlyMatchNullTerminator) {
  // Null terminator should match.
  ASSERT_STREQ(__llvm_libc::strrchr("", '\0'), "");
  // All other characters should not match.
  ASSERT_STREQ(__llvm_libc::strrchr("", 'A'), nullptr);
  ASSERT_STREQ(__llvm_libc::strrchr("", '2'), nullptr);
  ASSERT_STREQ(__llvm_libc::strrchr("", '*'), nullptr);
}

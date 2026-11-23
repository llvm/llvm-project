//===-- Unittests for wcsrchr ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/wchar_t.h"
#include "src/wchar/wcsrchr.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWCSRChrTest, FindsFirstCharacter) {
  // Should return pointer to original string since 'a' is the first character.
  const wchar_t *src = L"abcde";
  ASSERT_EQ(LIBC_NAMESPACE::wcsrchr(src, L'a'), src);
}

TEST(LlvmLibcWCSRChrTest, FindsMiddleCharacter) {
  // Should return pointer to 'c'.
  const wchar_t *src = L"abcde";
  ASSERT_EQ(LIBC_NAMESPACE::wcsrchr(src, L'c'), (src + 2));
}

TEST(LlvmLibcWCSRChrTest, FindsLastCharacterThatIsNotNullTerminator) {
  // Should return pointer to 'e'.
  const wchar_t *src = L"abcde";
  ASSERT_EQ(LIBC_NAMESPACE::wcsrchr(src, L'e'), (src + 4));
}

TEST(LlvmLibcWCSRChrTest, FindsNullTerminator) {
  // Should return pointer to null terminator.
  const wchar_t *src = L"abcde";
  ASSERT_EQ(LIBC_NAMESPACE::wcsrchr(src, L'\0'), (src + 5));
}

TEST(LlvmLibcWCSRChrTest, CharacterNotWithinStringShouldReturnNullptr) {
  // Since 'z' is not within the string, should return nullptr.
  const wchar_t *src = L"abcde";
  ASSERT_EQ(LIBC_NAMESPACE::wcsrchr(src, L'z'), nullptr);
}

TEST(LlvmLibcWCSRChrTest, ShouldFindLastOfDuplicates) {
  // Should return pointer to the last '1'.
  const wchar_t *src = L"abc1def1ghi";
  ASSERT_EQ((int)(LIBC_NAMESPACE::wcsrchr(src, L'1') - src), 7);

  // Should return pointer to the last 'X'
  const wchar_t *dups = L"XXXXX";
  ASSERT_EQ(LIBC_NAMESPACE::wcsrchr(dups, L'X'), dups + 4);
}

TEST(LlvmLibcWCSRChrTest, EmptyStringShouldOnlyMatchNullTerminator) {
  // Null terminator should match
  const wchar_t *src = L"";
  ASSERT_EQ(src, LIBC_NAMESPACE::wcsrchr(src, L'\0'));
  // All other characters should not match
  ASSERT_EQ(LIBC_NAMESPACE::wcsrchr(src, L'Z'), nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::wcsrchr(src, L'3'), nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::wcsrchr(src, L'*'), nullptr);
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST(LlvmLibcWCSRChrTest, NullptrCrash) {
  // Passing in a nullptr should crash the program.
  EXPECT_DEATH([] { LIBC_NAMESPACE::wcsrchr(nullptr, L'a'); }, WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS

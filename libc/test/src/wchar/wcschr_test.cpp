//===-- Unittests for wcschr ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/wchar_t.h"
#include "src/wchar/wcschr.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWCSChrTest, FindsFirstCharacter) {
  // Should return pointer to original string since 'a' is the first character.
  const wchar_t *src = L"abcde";
  ASSERT_EQ(LIBC_NAMESPACE::wcschr(src, L'a'), src);
}

TEST(LlvmLibcWCSChrTest, FindsMiddleCharacter) {
  // Should return pointer to 'c'.
  const wchar_t *src = L"abcde";
  ASSERT_EQ(LIBC_NAMESPACE::wcschr(src, L'c'), (src + 2));
}

TEST(LlvmLibcWCSChrTest, FindsLastCharacterThatIsNotNullTerminator) {
  // Should return pointer to 'e'.
  const wchar_t *src = L"abcde";
  ASSERT_EQ(LIBC_NAMESPACE::wcschr(src, L'e'), (src + 4));
}

TEST(LlvmLibcWCSChrTest, FindsNullTerminator) {
  // Should return pointer to null terminator.
  const wchar_t *src = L"abcde";
  ASSERT_EQ(LIBC_NAMESPACE::wcschr(src, L'\0'), (src + 5));
}

TEST(LlvmLibcWCSChrTest, CharacterNotWithinStringShouldReturnNullptr) {
  // Since 'z' is not within the string, should return nullptr.
  const wchar_t *src = L"abcde";
  ASSERT_EQ(LIBC_NAMESPACE::wcschr(src, L'z'), nullptr);
}

TEST(LlvmLibcWCSChrTest, ShouldFindFirstOfDuplicates) {
  // Should return pointer to the first '1'.
  const wchar_t *src = L"abc1def1ghi";
  ASSERT_EQ((int)(LIBC_NAMESPACE::wcschr(src, L'1') - src), 3);

  // Should return original string since 'X' is the first character.
  const wchar_t *dups = L"XXXXX";
  ASSERT_EQ(LIBC_NAMESPACE::wcschr(dups, L'X'), dups);
}

TEST(LlvmLibcWCSChrTest, EmptyStringShouldOnlyMatchNullTerminator) {
  // Null terminator should match
  const wchar_t *src = L"";
  ASSERT_EQ(src, LIBC_NAMESPACE::wcschr(src, L'\0'));
  // All other characters should not match
  ASSERT_EQ(LIBC_NAMESPACE::wcschr(src, L'Z'), nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::wcschr(src, L'3'), nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::wcschr(src, L'*'), nullptr);
}

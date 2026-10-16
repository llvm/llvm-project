//===-- Unittests for wcsncmp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcsncmp.h"
#include "test/UnitTest/Test.h"

// This group is just copies of the wcscmp tests, since all the same cases still
// need to be tested.

TEST(LlvmLibcWcsncmpTest, EmptyStringsShouldReturnZeroWithSufficientLength) {
  const wchar_t *s1 = L"";
  const wchar_t *s2 = L"";
  int result = LIBC_NAMESPACE::wcsncmp(s1, s2, 1);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::wcsncmp(s2, s1, 1);
  ASSERT_EQ(result, 0);
}

TEST(LlvmLibcWcsncmpTest,
     EmptyStringShouldNotEqualNonEmptyStringWithSufficientLength) {
  const wchar_t *empty = L"";
  const wchar_t *s2 = L"abc";
  int result = LIBC_NAMESPACE::wcsncmp(empty, s2, 3);
  ASSERT_LT(result, 0);

  // Similar case if empty string is second argument.
  const wchar_t *s3 = L"123";
  result = LIBC_NAMESPACE::wcsncmp(s3, empty, 3);
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcWcsncmpTest, EqualStringsShouldReturnZeroWithSufficientLength) {
  const wchar_t *s1 = L"abc";
  const wchar_t *s2 = L"abc";
  int result = LIBC_NAMESPACE::wcsncmp(s1, s2, 3);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::wcsncmp(s2, s1, 3);
  ASSERT_EQ(result, 0);
}

TEST(LlvmLibcWcsncmpTest,
     ShouldReturnResultOfFirstDifferenceWithSufficientLength) {
  const wchar_t *s1 = L"___B42__";
  const wchar_t *s2 = L"___C55__";
  int result = LIBC_NAMESPACE::wcsncmp(s1, s2, 8);
  ASSERT_LT(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::wcsncmp(s2, s1, 8);
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcWcsncmpTest,
     CapitalizedLetterShouldNotBeEqualWithSufficientLength) {
  const wchar_t *s1 = L"abcd";
  const wchar_t *s2 = L"abCd";
  int result = LIBC_NAMESPACE::wcsncmp(s1, s2, 4);
  ASSERT_GT(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::wcsncmp(s2, s1, 4);
  ASSERT_LT(result, 0);
}

TEST(LlvmLibcWcsncmpTest,
     UnequalLengthStringsShouldNotReturnZeroWithSufficientLength) {
  const wchar_t *s1 = L"abc";
  const wchar_t *s2 = L"abcd";
  int result = LIBC_NAMESPACE::wcsncmp(s1, s2, 4);
  ASSERT_LT(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::wcsncmp(s2, s1, 4);
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcWcsncmpTest, StringArgumentSwapChangesSignWithSufficientLength) {
  const wchar_t *a = L"a";
  const wchar_t *b = L"b";
  int result = LIBC_NAMESPACE::wcsncmp(b, a, 1);
  ASSERT_GT(result, 0);

  result = LIBC_NAMESPACE::wcsncmp(a, b, 1);
  ASSERT_LT(result, 0);
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST(LlvmLibcWcsncmpTest, NullptrCrash) {
  // Passing in a nullptr should crash the program.
  EXPECT_DEATH([] { LIBC_NAMESPACE::wcsncmp(L"aaaaaaaaaaaaaa", nullptr, 3); },
               WITH_SIGNAL(-1));
  EXPECT_DEATH([] { LIBC_NAMESPACE::wcsncmp(nullptr, L"aaaaaaaaaaaaaa", 3); },
               WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS

// This group is actually testing wcsncmp functionality

TEST(LlvmLibcWcsncmpTest, NonEqualStringsEqualWithLengthZero) {
  const wchar_t *s1 = L"abc";
  const wchar_t *s2 = L"def";
  int result = LIBC_NAMESPACE::wcsncmp(s1, s2, 0);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::wcsncmp(s2, s1, 0);
  ASSERT_EQ(result, 0);
}

TEST(LlvmLibcWcsncmpTest, NonEqualStringsNotEqualWithLengthOne) {
  const wchar_t *s1 = L"abc";
  const wchar_t *s2 = L"def";
  int result = LIBC_NAMESPACE::wcsncmp(s1, s2, 1);
  ASSERT_LT(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::wcsncmp(s2, s1, 1);
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcWcsncmpTest, NonEqualStringsEqualWithShorterLength) {
  const wchar_t *s1 = L"___B42__";
  const wchar_t *s2 = L"___C55__";
  int result = LIBC_NAMESPACE::wcsncmp(s1, s2, 3);
  ASSERT_EQ(result, 0);

  // This should return 'B' - 'C' = -1.
  result = LIBC_NAMESPACE::wcsncmp(s1, s2, 4);
  ASSERT_LT(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::wcsncmp(s2, s1, 3);
  ASSERT_EQ(result, 0);

  // This should return 'C' - 'B' = 1.
  result = LIBC_NAMESPACE::wcsncmp(s2, s1, 4);
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcWcsncmpTest, StringComparisonEndsOnNullByteEvenWithLongerLength) {
  const wchar_t *s1 = L"abc\0def";
  const wchar_t *s2 = L"abc\0abc";
  int result = LIBC_NAMESPACE::wcsncmp(s1, s2, 7);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::wcsncmp(s2, s1, 7);
  ASSERT_EQ(result, 0);
}

TEST(LlvmLibcWcsncmpTest, Case) {
  const wchar_t *s1 = L"aB";
  const wchar_t *s2 = L"ab";
  int result = LIBC_NAMESPACE::wcsncmp(s1, s2, 2);
  ASSERT_LT(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::wcsncmp(s2, s1, 2);
  ASSERT_GT(result, 0);
}

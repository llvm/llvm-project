//===-- Unittests for wcscoll ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "src/wchar/wcscoll.h"
#include "test/UnitTest/Test.h"

// TODO: Add more comprehensive tests once locale support is added.

TEST(LlvmLibcWcscollTest, EmptyStringsShouldReturnZero) {
  const wchar_t *s1 = L"";
  const wchar_t *s2 = L"";

  int result = LIBC_NAMESPACE::wcscoll(s1, s2);
  ASSERT_EQ(result, 0);
}

TEST(LlvmLibcWcscollTest, EmptyStringShouldNotEqualNonEmptyString) {
  const wchar_t *empty = L"";
  const wchar_t *s = L"abc";

  // An empty string comes before a non empty one lexicographically, so lt 0
  int result = LIBC_NAMESPACE::wcscoll(empty, s);
  ASSERT_LT(result, 0);

  // Check the reversed behaviour
  result = LIBC_NAMESPACE::wcscoll(s, empty);
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcWcscollTest, EqualStringsShouldReturnZero) {
  const wchar_t *s1 = L"abc";
  const wchar_t *s2 = L"abc";

  // Check if it returns 0 for two equal strings
  int result = LIBC_NAMESPACE::wcscoll(s1, s2);
  ASSERT_EQ(result, 0);

  // Verify for reversed operands
  result = LIBC_NAMESPACE::wcscoll(s2, s1);
  ASSERT_EQ(result, 0);
}

TEST(LlvmLibcWcscollTest, LexicographicalTest) {
  const wchar_t *s1 = L"abc";
  const wchar_t *s2 = L"def";

  // Check if it returns lt 0 for (abc, def)
  int result = LIBC_NAMESPACE::wcscoll(s1, s2);
  ASSERT_LT(result, 0);

  // Check if it returns gt 0 for (def, abc)
  result = LIBC_NAMESPACE::wcscoll(s2, s1);
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcWcscollTest, NonAsciiTest) {
  const wchar_t *s1 = L"AbCdEf__1230!! \u1111";
  const wchar_t *s2 = L"AbCdEf__1230!! \u1111\u2222";

  int result = LIBC_NAMESPACE::wcscoll(s1, s2);
  ASSERT_LT(result, 0);

  result = LIBC_NAMESPACE::wcscoll(s2, s1);
  ASSERT_GT(result, 0);

  result = LIBC_NAMESPACE::wcscoll(s1, s1);
  ASSERT_EQ(result, 0);

  // Empty string
  const wchar_t *empty = L"";
  result = LIBC_NAMESPACE::wcscoll(empty, s1);
  ASSERT_LT(result, 0);

  result = LIBC_NAMESPACE::wcscoll(s1, empty);
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcWcscollTest, EightDigitUCNTest) {
  const wchar_t *s1 = L"abC\U0001F44D"; // thumbs up emoji
  const wchar_t *s2 = L"abC\U0001F44E"; // thumbs down emoji

  int result = LIBC_NAMESPACE::wcscoll(s1, s2);
  ASSERT_LT(result, 0);

  result = LIBC_NAMESPACE::wcscoll(s2, s1);
  ASSERT_GT(result, 0);

  result = LIBC_NAMESPACE::wcscoll(s1, s1);
  ASSERT_EQ(result, 0);

  // empty string
  const wchar_t *empty = L"";
  result = LIBC_NAMESPACE::wcscoll(empty, s1);
  ASSERT_LT(result, 0);

  result = LIBC_NAMESPACE::wcscoll(s1, empty);
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcWcscollTest, AsciiVsNonAsciiTest) {
  const wchar_t *s1 = L"a";
  const wchar_t *s2 = L"\uFFFF";
  const wchar_t *s3 = L"\U0001000F";

  // ascii and 4 digit unicode
  int result = LIBC_NAMESPACE::wcscoll(s1, s2);
  ASSERT_LT(result, 0);

  result = LIBC_NAMESPACE::wcscoll(s2, s1);
  ASSERT_GT(result, 0);

  // ascii and 8 digit unicode
  result = LIBC_NAMESPACE::wcscoll(s1, s3);
  ASSERT_LT(result, 0);

  result = LIBC_NAMESPACE::wcscoll(s3, s1);
  ASSERT_GT(result, 0);

  // 4 digit unicode and 8 digit unicode
  result = LIBC_NAMESPACE::wcscoll(s2, s3);
  ASSERT_LT(result, 0);

  result = LIBC_NAMESPACE::wcscoll(s3, s2);
  ASSERT_GT(result, 0);
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST(LlvmLibcWcscollTest, NULLCheck) {
  // Passing in a nullptr should crash the program
  EXPECT_DEATH([] { LIBC_NAMESPACE::wcscoll(L"", nullptr); }, WITH_SIGNAL(-1));
  EXPECT_DEATH([] { LIBC_NAMESPACE::wcscoll(nullptr, L""); }, WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS

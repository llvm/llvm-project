//===-- Unittests for wcscmp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcscmp.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWcscmpTest, EmptyStringsShouldReturnZero) {
  const wchar_t *s1 = L"";
  const wchar_t *s2 = L"";
  int result = LIBC_NAMESPACE::wcscmp(s1, s2);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::wcscmp(s2, s1);
  ASSERT_EQ(result, 0);
}

TEST(LlvmLibcWcscmpTest, EmptyStringShouldNotEqualNonEmptyString) {
  const wchar_t *empty = L"";
  const wchar_t *s2 = L"abc";
  int result = LIBC_NAMESPACE::wcscmp(empty, s2);
  ASSERT_LT(result, 0);

  // Similar case if empty string is second argument.
  const wchar_t *s3 = L"123";
  result = LIBC_NAMESPACE::wcscmp(s3, empty);
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcWcscmpTest, EqualStringsShouldReturnZero) {
  const wchar_t *s1 = L"abc";
  const wchar_t *s2 = L"abc";
  int result = LIBC_NAMESPACE::wcscmp(s1, s2);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::wcscmp(s2, s1);
  ASSERT_EQ(result, 0);
}

TEST(LlvmLibcWcscmpTest, ShouldReturnResultOfFirstDifference) {
  const wchar_t *s1 = L"___B42__";
  const wchar_t *s2 = L"___C55__";
  int result = LIBC_NAMESPACE::wcscmp(s1, s2);
  ASSERT_LT(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::wcscmp(s2, s1);
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcWcscmpTest, CapitalizedLetterShouldNotBeEqual) {
  const wchar_t *s1 = L"abcd";
  const wchar_t *s2 = L"abCd";
  int result = LIBC_NAMESPACE::wcscmp(s1, s2);
  ASSERT_GT(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::wcscmp(s2, s1);
  ASSERT_LT(result, 0);
}

TEST(LlvmLibcWcscmpTest, UnequalLengthStringsShouldNotReturnZero) {
  const wchar_t *s1 = L"abc";
  const wchar_t *s2 = L"abcd";
  int result = LIBC_NAMESPACE::wcscmp(s1, s2);
  ASSERT_LT(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::wcscmp(s2, s1);
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcWcscmpTest, StringArgumentSwapChangesSign) {
  const wchar_t *a = L"a";
  const wchar_t *b = L"b";
  int result = LIBC_NAMESPACE::wcscmp(b, a);
  ASSERT_GT(result, 0);

  result = LIBC_NAMESPACE::wcscmp(a, b);
  ASSERT_LT(result, 0);
}

#if defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_HAS_SANITIZER)
TEST(LlvmLibcWcscmpTest, NullptrCrash) {
  // Passing in a nullptr should crash the program.
  EXPECT_DEATH([] { LIBC_NAMESPACE::wcscmp(L"aaaaaaaaaaaaaa", nullptr); },
               WITH_SIGNAL(-1));
  EXPECT_DEATH([] { LIBC_NAMESPACE::wcscmp(nullptr, L"aaaaaaaaaaaaaa"); },
               WITH_SIGNAL(-1));
}
#endif // LIBC_HAS_ADDRESS_SANITIZER

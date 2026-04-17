//===-- Unittests for strcmp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcmp.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrCmpTest, EmptyStringsShouldReturnZero) {
  const char *s1 = "";
  const char *s2 = "";
  int result = LIBC_NAMESPACE::strcmp(s1, s2);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::strcmp(s2, s1);
  ASSERT_EQ(result, 0);
}

TEST(LlvmLibcStrCmpTest, EmptyStringShouldNotEqualNonEmptyString) {
  const char *empty = "";
  const char *s2 = "abc";
  int result = LIBC_NAMESPACE::strcmp(empty, s2);
  ASSERT_LT(result, 0);

  // Similar case if empty string is second argument.
  const char *s3 = "123";
  result = LIBC_NAMESPACE::strcmp(s3, empty);
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcStrCmpTest, EqualStringsShouldReturnZero) {
  const char *s1 = "abc";
  const char *s2 = "abc";
  int result = LIBC_NAMESPACE::strcmp(s1, s2);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::strcmp(s2, s1);
  ASSERT_EQ(result, 0);
}

TEST(LlvmLibcStrCmpTest, ShouldReturnResultOfFirstDifference) {
  const char *s1 = "___B42__";
  const char *s2 = "___C55__";
  int result = LIBC_NAMESPACE::strcmp(s1, s2);
  ASSERT_LT(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::strcmp(s2, s1);
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcStrCmpTest, CapitalizedLetterShouldNotBeEqual) {
  const char *s1 = "abcd";
  const char *s2 = "abCd";
  int result = LIBC_NAMESPACE::strcmp(s1, s2);
  ASSERT_GT(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::strcmp(s2, s1);
  ASSERT_LT(result, 0);
}

TEST(LlvmLibcStrCmpTest, UnequalLengthStringsShouldNotReturnZero) {
  const char *s1 = "abc";
  const char *s2 = "abcd";
  int result = LIBC_NAMESPACE::strcmp(s1, s2);
  ASSERT_LT(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::strcmp(s2, s1);
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcStrCmpTest, StringArgumentSwapChangesSign) {
  const char *a = "a";
  const char *b = "b";
  int result = LIBC_NAMESPACE::strcmp(b, a);
  ASSERT_GT(result, 0);

  result = LIBC_NAMESPACE::strcmp(a, b);
  ASSERT_LT(result, 0);
}

TEST(LlvmLibcStrCmpTest, Case) {
  const char *s1 = "aB";
  const char *s2 = "ab";
  int result = LIBC_NAMESPACE::strcmp(s1, s2);
  ASSERT_LT(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::strcmp(s2, s1);
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcStrCmpTest, CharactersGreaterThan127ShouldBePositive) {
  const char s1[] = {static_cast<char>(128), '\0'};
  const char s2[] = {'\0'};
  int result = LIBC_NAMESPACE::strcmp(s1, s2);
  ASSERT_GT(result, 0);
}

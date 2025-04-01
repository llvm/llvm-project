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
  // This should be '\0' - 'a' = -97
  ASSERT_EQ(result, '\0' - 'a');

  // Similar case if empty string is second argument.
  const char *s3 = "123";
  result = LIBC_NAMESPACE::strcmp(s3, empty);
  // This should be '1' - '\0' = 49
  ASSERT_EQ(result, '1' - '\0');
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
  // This should return 'B' - 'C' = -1.
  ASSERT_EQ(result, 'B' - 'C');

  // Verify operands reversed.
  result = LIBC_NAMESPACE::strcmp(s2, s1);
  // This should return 'C' - 'B' = 1.
  ASSERT_EQ(result, 'C' - 'B');
}

TEST(LlvmLibcStrCmpTest, CapitalizedLetterShouldNotBeEqual) {
  const char *s1 = "abcd";
  const char *s2 = "abCd";
  int result = LIBC_NAMESPACE::strcmp(s1, s2);
  // 'c' - 'C' = 32.
  ASSERT_EQ(result, 'c' - 'C');

  // Verify operands reversed.
  result = LIBC_NAMESPACE::strcmp(s2, s1);
  // 'C' - 'c' = -32.
  ASSERT_EQ(result, 'C' - 'c');
}

TEST(LlvmLibcStrCmpTest, UnequalLengthStringsShouldNotReturnZero) {
  const char *s1 = "abc";
  const char *s2 = "abcd";
  int result = LIBC_NAMESPACE::strcmp(s1, s2);
  // '\0' - 'd' = -100.
  ASSERT_EQ(result, -'\0' - 'd');

  // Verify operands reversed.
  result = LIBC_NAMESPACE::strcmp(s2, s1);
  // 'd' - '\0' = 100.
  ASSERT_EQ(result, 'd' - '\0');
}

TEST(LlvmLibcStrCmpTest, StringArgumentSwapChangesSign) {
  const char *a = "a";
  const char *b = "b";
  int result = LIBC_NAMESPACE::strcmp(b, a);
  // 'b' - 'a' = 1.
  ASSERT_EQ(result, 'b' - 'a');

  result = LIBC_NAMESPACE::strcmp(a, b);
  // 'a' - 'b' = -1.
  ASSERT_EQ(result, 'a' - 'b');
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

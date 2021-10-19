//===-- Unittests for strncmp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strncmp.h"
#include "utils/UnitTest/Test.h"

// This group is just copies of the strcmp tests, since all the same cases still
// need to be tested.

TEST(LlvmLibcStrNCmpTest, EmptyStringsShouldReturnZeroWithSufficientLength) {
  const char *s1 = "";
  const char *s2 = "";
  int result = __llvm_libc::strncmp(s1, s2, 1);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = __llvm_libc::strncmp(s2, s1, 1);
  ASSERT_EQ(result, 0);
}

TEST(LlvmLibcStrNCmpTest,
     EmptyStringShouldNotEqualNonEmptyStringWithSufficientLength) {
  const char *empty = "";
  const char *s2 = "abc";
  int result = __llvm_libc::strncmp(empty, s2, 3);
  // This should be '\0' - 'a' = -97
  ASSERT_EQ(result, -97);

  // Similar case if empty string is second argument.
  const char *s3 = "123";
  result = __llvm_libc::strncmp(s3, empty, 3);
  // This should be '1' - '\0' = 49
  ASSERT_EQ(result, 49);
}

TEST(LlvmLibcStrNCmpTest, EqualStringsShouldReturnZeroWithSufficientLength) {
  const char *s1 = "abc";
  const char *s2 = "abc";
  int result = __llvm_libc::strncmp(s1, s2, 3);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = __llvm_libc::strncmp(s2, s1, 3);
  ASSERT_EQ(result, 0);
}

TEST(LlvmLibcStrNCmpTest,
     ShouldReturnResultOfFirstDifferenceWithSufficientLength) {
  const char *s1 = "___B42__";
  const char *s2 = "___C55__";
  int result = __llvm_libc::strncmp(s1, s2, 8);
  // This should return 'B' - 'C' = -1.
  ASSERT_EQ(result, -1);

  // Verify operands reversed.
  result = __llvm_libc::strncmp(s2, s1, 8);
  // This should return 'C' - 'B' = 1.
  ASSERT_EQ(result, 1);
}

TEST(LlvmLibcStrNCmpTest,
     CapitalizedLetterShouldNotBeEqualWithSufficientLength) {
  const char *s1 = "abcd";
  const char *s2 = "abCd";
  int result = __llvm_libc::strncmp(s1, s2, 4);
  // 'c' - 'C' = 32.
  ASSERT_EQ(result, 32);

  // Verify operands reversed.
  result = __llvm_libc::strncmp(s2, s1, 4);
  // 'C' - 'c' = -32.
  ASSERT_EQ(result, -32);
}

TEST(LlvmLibcStrNCmpTest,
     UnequalLengthStringsShouldNotReturnZeroWithSufficientLength) {
  const char *s1 = "abc";
  const char *s2 = "abcd";
  int result = __llvm_libc::strncmp(s1, s2, 4);
  // '\0' - 'd' = -100.
  ASSERT_EQ(result, -100);

  // Verify operands reversed.
  result = __llvm_libc::strncmp(s2, s1, 4);
  // 'd' - '\0' = 100.
  ASSERT_EQ(result, 100);
}

TEST(LlvmLibcStrNCmpTest, StringArgumentSwapChangesSignWithSufficientLength) {
  const char *a = "a";
  const char *b = "b";
  int result = __llvm_libc::strncmp(b, a, 1);
  // 'b' - 'a' = 1.
  ASSERT_EQ(result, 1);

  result = __llvm_libc::strncmp(a, b, 1);
  // 'a' - 'b' = -1.
  ASSERT_EQ(result, -1);
}

// This group is actually testing strncmp functionality

TEST(LlvmLibcStrNCmpTest, NonEqualStringsEqualWithLengthZero) {
  const char *s1 = "abc";
  const char *s2 = "def";
  int result = __llvm_libc::strncmp(s1, s2, 0);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = __llvm_libc::strncmp(s2, s1, 0);
  ASSERT_EQ(result, 0);
}

TEST(LlvmLibcStrNCmpTest, NonEqualStringsNotEqualWithLengthOne) {
  const char *s1 = "abc";
  const char *s2 = "def";
  int result = __llvm_libc::strncmp(s1, s2, 1);
  ASSERT_EQ(result, -3);

  // Verify operands reversed.
  result = __llvm_libc::strncmp(s2, s1, 1);
  ASSERT_EQ(result, 3);
}

TEST(LlvmLibcStrNCmpTest, NonEqualStringsEqualWithShorterLength) {
  const char *s1 = "___B42__";
  const char *s2 = "___C55__";
  int result = __llvm_libc::strncmp(s1, s2, 3);
  ASSERT_EQ(result, 0);

  // This should return 'B' - 'C' = -1.
  result = __llvm_libc::strncmp(s1, s2, 4);
  ASSERT_EQ(result, -1);

  // Verify operands reversed.
  result = __llvm_libc::strncmp(s2, s1, 3);
  ASSERT_EQ(result, 0);

  // This should return 'C' - 'B' = 1.
  result = __llvm_libc::strncmp(s2, s1, 4);
  ASSERT_EQ(result, 1);
}

TEST(LlvmLibcStrNCmpTest, StringComparisonEndsOnNullByteEvenWithLongerLength) {
  const char *s1 = "abc\0def";
  const char *s2 = "abc\0abc";
  int result = __llvm_libc::strncmp(s1, s2, 7);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = __llvm_libc::strncmp(s2, s1, 7);
  ASSERT_EQ(result, 0);
}

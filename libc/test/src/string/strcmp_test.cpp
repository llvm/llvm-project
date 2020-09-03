//===-- Unittests for strcmp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcmp.h"
#include "utils/UnitTest/Test.h"

TEST(StrCmpTest, EmptyStringsShouldReturnZero) {
  const char *s1 = "";
  const char *s2 = "";
  int result = __llvm_libc::strcmp(s1, s2);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = __llvm_libc::strcmp(s2, s1);
  ASSERT_EQ(result, 0);
}

TEST(StrCmpTest, EmptyStringShouldNotEqualNonEmptyString) {
  const char *empty = "";
  const char *s2 = "abc";
  int result = __llvm_libc::strcmp(empty, s2);
  // This should be '\0' - 'a' = -97
  ASSERT_EQ(result, -97);

  // Similar case if empty string is second argument.
  const char *s3 = "123";
  result = __llvm_libc::strcmp(s3, empty);
  // This should be '1' - '\0' = 49
  ASSERT_EQ(result, 49);
}

TEST(StrCmpTest, EqualStringsShouldReturnZero) {
  const char *s1 = "abc";
  const char *s2 = "abc";
  int result = __llvm_libc::strcmp(s1, s2);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = __llvm_libc::strcmp(s2, s1);
  ASSERT_EQ(result, 0);
}

TEST(StrCmpTest, ShouldReturnResultOfFirstDifference) {
  const char *s1 = "___B42__";
  const char *s2 = "___C55__";
  int result = __llvm_libc::strcmp(s1, s2);
  // This should return 'B' - 'C' = -1.
  ASSERT_EQ(result, -1);

  // Verify operands reversed.
  result = __llvm_libc::strcmp(s2, s1);
  // This should return 'C' - 'B' = 1.
  ASSERT_EQ(result, 1);
}

TEST(StrCmpTest, CapitalizedLetterShouldNotBeEqual) {
  const char *s1 = "abcd";
  const char *s2 = "abCd";
  int result = __llvm_libc::strcmp(s1, s2);
  // 'c' - 'C' = 32.
  ASSERT_EQ(result, 32);

  // Verify operands reversed.
  result = __llvm_libc::strcmp(s2, s1);
  // 'C' - 'c' = -32.
  ASSERT_EQ(result, -32);
}

TEST(StrCmpTest, UnequalLengthStringsShouldNotReturnZero) {
  const char *s1 = "abc";
  const char *s2 = "abcd";
  int result = __llvm_libc::strcmp(s1, s2);
  // '\0' - 'd' = -100.
  ASSERT_EQ(result, -100);

  // Verify operands reversed.
  result = __llvm_libc::strcmp(s2, s1);
  // 'd' - '\0' = 100.
  ASSERT_EQ(result, 100);
}

TEST(StrCmpTest, StringArgumentSwapChangesSign) {
  const char *a = "a";
  const char *b = "b";
  int result = __llvm_libc::strcmp(b, a);
  // 'b' - 'a' = 1.
  ASSERT_EQ(result, 1);

  result = __llvm_libc::strcmp(a, b);
  // 'a' - 'b' = -1.
  ASSERT_EQ(result, -1);
}

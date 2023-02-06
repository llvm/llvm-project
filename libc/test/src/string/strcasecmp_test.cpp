//===-- Unittests for strcasecmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcasecmp.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrCaseCmpTest, EmptyStringsShouldReturnZero) {
  const char *s1 = "";
  const char *s2 = "";
  int result = __llvm_libc::strcasecmp(s1, s2);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = __llvm_libc::strcasecmp(s2, s1);
  ASSERT_EQ(result, 0);
}

TEST(LlvmLibcStrCaseCmpTest, EmptyStringShouldNotEqualNonEmptyString) {
  const char *empty = "";
  const char *s2 = "abc";
  int result = __llvm_libc::strcasecmp(empty, s2);
  // This should be '\0' - 'a' = -97
  ASSERT_LT(result, 0);

  // Similar case if empty string is second argument.
  const char *s3 = "123";
  result = __llvm_libc::strcasecmp(s3, empty);
  // This should be '1' - '\0' = 49
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcStrCaseCmpTest, Case) {
  const char *s1 = "aB";
  const char *s2 = "ab";
  int result = __llvm_libc::strcasecmp(s1, s2);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = __llvm_libc::strcasecmp(s2, s1);
  ASSERT_EQ(result, 0);
}

//===-- Unittests for strncasecmp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/strings/strncasecmp.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrNCaseCmpTest,
     EmptyStringsShouldReturnZeroWithSufficientLength) {
  const char *s1 = "";
  const char *s2 = "";
  int result = LIBC_NAMESPACE::strncasecmp(s1, s2, 1);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::strncasecmp(s2, s1, 1);
  ASSERT_EQ(result, 0);
}

TEST(LlvmLibcStrNCaseCmpTest,
     EmptyStringShouldNotEqualNonEmptyStringWithSufficientLength) {
  const char *empty = "";
  const char *s2 = "abc";
  int result = LIBC_NAMESPACE::strncasecmp(empty, s2, 3);
  // This should be '\0' - 'a' = -97
  ASSERT_LT(result, 0);

  // Similar case if empty string is second argument.
  const char *s3 = "123";
  result = LIBC_NAMESPACE::strncasecmp(s3, empty, 3);
  // This should be '1' - '\0' = 49
  ASSERT_GT(result, 0);
}

TEST(LlvmLibcStrNCaseCmpTest, Case) {
  const char *s1 = "aB";
  const char *s2 = "ab";
  int result = LIBC_NAMESPACE::strncasecmp(s1, s2, 2);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::strncasecmp(s2, s1, 2);
  ASSERT_EQ(result, 0);
}

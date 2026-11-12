//===-- Unittests for strxfrm ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strxfrm.h"
#include "test/UnitTest/Test.h"

#include "src/string/string_utils.h"

// TODO: Add more comprehensive tests once locale support is added.

TEST(LlvmLibcStrxfrmTest, SimpleTestSufficientlySizedN) {
  const char *src = "abc";
  const size_t n = 5;

  char dest[n];
  size_t result = LIBC_NAMESPACE::strxfrm(dest, src, n);
  ASSERT_EQ(result, LIBC_NAMESPACE::internal::string_length(src));
  ASSERT_STREQ(dest, src);
}

TEST(LlvmLibcStrxfrmTest, SimpleTestExactSizedN) {
  const char *src = "abc";
  const size_t n = 4;

  char dest[n];
  size_t result = LIBC_NAMESPACE::strxfrm(dest, src, n);
  ASSERT_EQ(result, LIBC_NAMESPACE::internal::string_length(src));
  ASSERT_STREQ(dest, src);
}

TEST(LlvmLibcStrxfrmTest, SimpleTestInsufficientlySizedN) {
  const char *src = "abc";
  const size_t n = 3;

  // Verify strxfrm does not modify dest if src len >= n
  char dest[n] = {'x', 'x', '\0'};
  size_t result = LIBC_NAMESPACE::strxfrm(dest, src, n);
  ASSERT_GE(result, n);
  ASSERT_STREQ(dest, "xx");
}

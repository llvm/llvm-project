//===-- Unittests for strdup ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/string/strdup.h"
#include "test/UnitTest/Test.h"

#include <stdlib.h>

TEST(LlvmLibcStrDupTest, EmptyString) {
  const char *empty = "";

  LIBC_NAMESPACE::libc_errno = 0;
  char *result = LIBC_NAMESPACE::strdup(empty);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_NE(result, static_cast<char *>(nullptr));
  ASSERT_NE(empty, const_cast<const char *>(result));
  ASSERT_STREQ(empty, result);
  ::free(result);
}

TEST(LlvmLibcStrDupTest, AnyString) {
  const char *abc = "abc";

  LIBC_NAMESPACE::libc_errno = 0;
  char *result = LIBC_NAMESPACE::strdup(abc);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_NE(result, static_cast<char *>(nullptr));
  ASSERT_NE(abc, const_cast<const char *>(result));
  ASSERT_STREQ(abc, result);
  ::free(result);
}

TEST(LlvmLibcStrDupTest, NullPtr) {
  LIBC_NAMESPACE::libc_errno = 0;
  char *result = LIBC_NAMESPACE::strdup(nullptr);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_EQ(result, static_cast<char *>(nullptr));
}

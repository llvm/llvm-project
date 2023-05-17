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

  libc_errno = 0;
  char *result = __llvm_libc::strdup(empty);
  ASSERT_EQ(libc_errno, 0);

  ASSERT_NE(result, static_cast<char *>(nullptr));
  ASSERT_NE(empty, const_cast<const char *>(result));
  ASSERT_STREQ(empty, result);
  ::free(result);
}

TEST(LlvmLibcStrDupTest, AnyString) {
  const char *abc = "abc";

  libc_errno = 0;
  char *result = __llvm_libc::strdup(abc);
  ASSERT_EQ(libc_errno, 0);

  ASSERT_NE(result, static_cast<char *>(nullptr));
  ASSERT_NE(abc, const_cast<const char *>(result));
  ASSERT_STREQ(abc, result);
  ::free(result);
}

TEST(LlvmLibcStrDupTest, NullPtr) {
  libc_errno = 0;
  char *result = __llvm_libc::strdup(nullptr);
  ASSERT_EQ(libc_errno, 0);

  ASSERT_EQ(result, static_cast<char *>(nullptr));
}

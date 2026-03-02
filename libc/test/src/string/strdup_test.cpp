//===-- Unittests for strdup ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strdup.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcStrDupTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcStrDupTest, EmptyString) {
  const char *empty = "";

  char *result = LIBC_NAMESPACE::strdup(empty);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_NE(result, static_cast<char *>(nullptr));
  ASSERT_NE(empty, const_cast<const char *>(result));
  ASSERT_STREQ(empty, result);
  ::free(result);
}

TEST_F(LlvmLibcStrDupTest, AnyString) {
  const char *abc = "abc";

  char *result = LIBC_NAMESPACE::strdup(abc);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_NE(result, static_cast<char *>(nullptr));
  ASSERT_NE(abc, const_cast<const char *>(result));
  ASSERT_STREQ(abc, result);
  ::free(result);
}

TEST_F(LlvmLibcStrDupTest, NullPtr) {
  char *result = LIBC_NAMESPACE::strdup(nullptr);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_EQ(result, static_cast<char *>(nullptr));
}

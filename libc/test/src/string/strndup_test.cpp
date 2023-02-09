//===-- Unittests for strndup ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strndup.h"
#include "test/UnitTest/Test.h"
#include <stdlib.h>

TEST(LlvmLibcstrndupTest, EmptyString) {
  const char *empty = "";

  char *result = __llvm_libc::strndup(empty, 1);
  ASSERT_NE(result, static_cast<char *>(nullptr));
  ASSERT_NE(empty, const_cast<const char *>(result));
  ASSERT_STREQ(empty, result);
  ::free(result);
}

TEST(LlvmLibcstrndupTest, AnyString) {
  const char *abc = "abc";

  char *result = __llvm_libc::strndup(abc, 3);

  ASSERT_NE(result, static_cast<char *>(nullptr));
  ASSERT_NE(abc, const_cast<const char *>(result));
  ASSERT_STREQ(abc, result);
  ::free(result);

  result = __llvm_libc::strndup(abc, 1);

  ASSERT_NE(result, static_cast<char *>(nullptr));
  ASSERT_NE(abc, const_cast<const char *>(result));
  ASSERT_STREQ("a", result);
  ::free(result);

  result = __llvm_libc::strndup(abc, 10);

  ASSERT_NE(result, static_cast<char *>(nullptr));
  ASSERT_NE(abc, const_cast<const char *>(result));
  ASSERT_STREQ(abc, result);
  ::free(result);
}

TEST(LlvmLibcstrndupTest, NullPtr) {
  char *result = __llvm_libc::strndup(nullptr, 0);

  ASSERT_EQ(result, static_cast<char *>(nullptr));
}

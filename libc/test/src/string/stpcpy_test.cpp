//===-- Unittests for stpcpy ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/stpcpy.h"
#include "test/UnitTest/Test.h"

#include "src/string/string_utils.h"

TEST(LlvmLibcStpCpyTest, EmptySrc) {
  const char *empty = "";
  size_t src_size = LIBC_NAMESPACE::internal::string_length(empty);
  char dest[4] = {'a', 'b', 'c', '\0'};

  char *result = LIBC_NAMESPACE::stpcpy(dest, empty);
  ASSERT_EQ(dest + src_size, result);
  ASSERT_EQ(result[0], '\0');
  ASSERT_STREQ(dest, empty);
}

TEST(LlvmLibcStpCpyTest, EmptyDest) {
  const char *abc = "abc";
  size_t src_size = LIBC_NAMESPACE::internal::string_length(abc);
  char dest[4];

  char *result = LIBC_NAMESPACE::stpcpy(dest, abc);
  ASSERT_EQ(dest + src_size, result);
  ASSERT_EQ(result[0], '\0');
  ASSERT_STREQ(dest, abc);
}

TEST(LlvmLibcStpCpyTest, OffsetDest) {
  const char *abc = "abc";
  size_t src_size = LIBC_NAMESPACE::internal::string_length(abc);
  char dest[7] = {'x', 'y', 'z'};

  char *result = LIBC_NAMESPACE::stpcpy(dest + 3, abc);
  ASSERT_EQ(dest + 3 + src_size, result);
  ASSERT_EQ(result[0], '\0');
  ASSERT_STREQ(dest, "xyzabc");
}

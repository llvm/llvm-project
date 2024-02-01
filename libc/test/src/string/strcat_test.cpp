//===-- Unittests for strcat ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcat.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrCatTest, EmptyDest) {
  const char *abc = "abc";
  char dest[4];

  dest[0] = '\0';

  char *result = LIBC_NAMESPACE::strcat(dest, abc);
  ASSERT_EQ(dest, result);
  ASSERT_STREQ(dest, result);
  ASSERT_STREQ(dest, abc);
}

TEST(LlvmLibcStrCatTest, NonEmptyDest) {
  const char *abc = "abc";
  char dest[7];

  dest[0] = 'x';
  dest[1] = 'y';
  dest[2] = 'z';
  dest[3] = '\0';

  char *result = LIBC_NAMESPACE::strcat(dest, abc);
  ASSERT_EQ(dest, result);
  ASSERT_STREQ(dest, result);
  ASSERT_STREQ(dest, "xyzabc");
}

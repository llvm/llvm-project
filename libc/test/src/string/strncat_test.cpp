//===-- Unittests for strncat ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strncat.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrNCatTest, EmptyDest) {
  const char *abc = "abc";
  char dest[4];

  dest[0] = '\0';

  // Start by copying nothing
  char *result = LIBC_NAMESPACE::strncat(dest, abc, 0);
  ASSERT_EQ(dest, result);
  ASSERT_EQ(dest[0], '\0');

  // Then copy part of it.
  result = LIBC_NAMESPACE::strncat(dest, abc, 1);
  ASSERT_EQ(dest, result);
  ASSERT_STREQ(dest, "a");

  // Reset for the last test.
  dest[0] = '\0';

  // Then copy all of it.
  result = LIBC_NAMESPACE::strncat(dest, abc, 3);
  ASSERT_EQ(dest, result);
  ASSERT_STREQ(dest, result);
  ASSERT_STREQ(dest, abc);
}

TEST(LlvmLibcStrNCatTest, NonEmptyDest) {
  const char *abc = "abc";
  char dest[7];

  dest[0] = 'x';
  dest[1] = 'y';
  dest[2] = 'z';
  dest[3] = '\0';

  // Copy only part of the string onto the end
  char *result = LIBC_NAMESPACE::strncat(dest, abc, 1);
  ASSERT_EQ(dest, result);
  ASSERT_STREQ(dest, "xyza");

  // Copy a bit more, but without resetting.
  result = LIBC_NAMESPACE::strncat(dest, abc, 2);
  ASSERT_EQ(dest, result);
  ASSERT_STREQ(dest, "xyzaab");

  // Set just the end marker, to make sure it overwrites properly.
  dest[3] = '\0';

  result = LIBC_NAMESPACE::strncat(dest, abc, 3);
  ASSERT_EQ(dest, result);
  ASSERT_STREQ(dest, "xyzabc");

  // Check that copying still works when count > src length
  dest[0] = '\0';
  // And that it doesn't write beyond what is necessary.
  dest[4] = 'Z';
  result = LIBC_NAMESPACE::strncat(dest, abc, 4);
  ASSERT_EQ(dest, result);
  ASSERT_STREQ(dest, "abc");
  ASSERT_EQ(dest[4], 'Z');

  result = LIBC_NAMESPACE::strncat(dest, abc, 5);
  ASSERT_EQ(dest, result);
  ASSERT_STREQ(dest, "abcabc");
}

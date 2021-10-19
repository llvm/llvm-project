//===-- Unittests for strcpy ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcpy.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcStrCpyTest, EmptyDest) {
  const char *abc = "abc";
  char dest[4];

  char *result = __llvm_libc::strcpy(dest, abc);
  ASSERT_EQ(dest, result);
  ASSERT_STREQ(dest, result);
  ASSERT_STREQ(dest, abc);
}

TEST(LlvmLibcStrCpyTest, OffsetDest) {
  const char *abc = "abc";
  char dest[7];

  dest[0] = 'x';
  dest[1] = 'y';
  dest[2] = 'z';

  char *result = __llvm_libc::strcpy(dest + 3, abc);
  ASSERT_EQ(dest + 3, result);
  ASSERT_STREQ(dest + 3, result);
  ASSERT_STREQ(dest, "xyzabc");
}

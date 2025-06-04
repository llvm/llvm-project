//===-- Unittests for strcpy ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "src/string/strcpy.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrCpyTest, EmptySrc) {
  const char *empty = "";
  char dest[4] = {'a', 'b', 'c', '\0'};

  char *result = LIBC_NAMESPACE::strcpy(dest, empty);
  ASSERT_EQ(dest, result);
  ASSERT_STREQ(dest, result);
  ASSERT_STREQ(dest, empty);
}

TEST(LlvmLibcStrCpyTest, EmptyDest) {
  const char *abc = "abc";
  char dest[4];

  char *result = LIBC_NAMESPACE::strcpy(dest, abc);
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

  char *result = LIBC_NAMESPACE::strcpy(dest + 3, abc);
  ASSERT_EQ(dest + 3, result);
  ASSERT_STREQ(dest + 3, result);
  ASSERT_STREQ(dest, "xyzabc");
}

#if defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_HAS_SANITIZER)

TEST(LlvmLibcStrCpyTest, CrashOnNullPtr) {
  ASSERT_DEATH([]() { LIBC_NAMESPACE::strcpy(nullptr, nullptr); },
               WITH_SIGNAL(-1));
}

#endif // defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_HAS_SANITIZER)

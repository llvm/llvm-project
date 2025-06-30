//===-- Unittests for strcoll ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "src/string/strcoll.h"
#include "test/UnitTest/Test.h"

// TODO: Add more comprehensive tests once locale support is added.

TEST(LlvmLibcStrcollTest, SimpleTest) {
  const char *s1 = "abc";
  const char *s2 = "abc";
  const char *s3 = "def";
  int result = LIBC_NAMESPACE::strcoll(s1, s2);
  ASSERT_EQ(result, 0);

  // Verify operands reversed.
  result = LIBC_NAMESPACE::strcoll(s2, s1);
  ASSERT_EQ(result, 0);

  result = LIBC_NAMESPACE::strcoll(s1, s3);
  ASSERT_LT(result, 0);

  result = LIBC_NAMESPACE::strcoll(s3, s1);
  ASSERT_GT(result, 0);
}

#if defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_HAS_SANITIZER)

TEST(LlvmLibcStrcollTest, CrashOnNullPtr) {
  ASSERT_DEATH([]() { LIBC_NAMESPACE::strcoll(nullptr, nullptr); },
               WITH_SIGNAL(-1));
}

#endif // defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_HAS_SANITIZER)

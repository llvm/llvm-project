//===-- Unittests for strlcpy ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strlcpy.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrlcpyTest, TooBig) {
  const char *str = "abc";
  char buf[2];
  EXPECT_EQ(LIBC_NAMESPACE::strlcpy(buf, str, 2), size_t(3));
  EXPECT_STREQ(buf, "a");

  EXPECT_EQ(LIBC_NAMESPACE::strlcpy(nullptr, str, 0), size_t(3));
}

TEST(LlvmLibcStrlcpyTest, Smaller) {
  const char *str = "abc";
  char buf[7]{"111111"};

  EXPECT_EQ(LIBC_NAMESPACE::strlcpy(buf, str, 7), size_t(3));
  EXPECT_STREQ(buf, "abc");
  EXPECT_STREQ(buf + 4, "11");
}

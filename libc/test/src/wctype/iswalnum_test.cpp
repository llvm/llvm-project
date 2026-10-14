//===-- Unittests for iswalnum --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/iswalnum.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibciswalnum, SimpleTest) {
  EXPECT_NE(LIBC_NAMESPACE::iswalnum('A'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswalnum('z'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswalnum('0'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswalnum(' '), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswalnum('?'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswalnum('\0'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswalnum(-1), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswalnum('a'), 1);
  EXPECT_EQ(LIBC_NAMESPACE::iswalnum('Z'), 1);
  EXPECT_EQ(LIBC_NAMESPACE::iswalnum('9'), 1);
}

//===-- Unittests for iswspace --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/iswspace.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibciswspace, SimpleTest) {
  EXPECT_NE(LIBC_NAMESPACE::iswspace(' '), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswspace('\t'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswspace('\n'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswspace('\v'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswspace('\f'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswspace('\r'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswspace('a'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswspace('3'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswspace('?'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswspace('\0'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswspace(-1), 0);
}

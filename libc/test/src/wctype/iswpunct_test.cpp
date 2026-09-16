//===-- Unittests for iswpunct --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/iswpunct.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibciswpunct, SimpleTest) {
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('!'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('\"'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('#'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('$'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('%'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('&'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('\''), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('('), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct(')'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('*'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('+'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct(','), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('-'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('.'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('/'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct(':'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct(';'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('<'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('='), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('>'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('?'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('@'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('['), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('\\'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct(']'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('^'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('_'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('`'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('{'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('|'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('}'), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswpunct('~'), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswpunct('\n'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswpunct('\v'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswpunct('\f'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswpunct('\r'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswpunct('\0'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswpunct('\t'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswpunct(' '), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswpunct('0'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswpunct('5'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswpunct('9'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswpunct('A'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswpunct('M'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswpunct('Z'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswpunct('a'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswpunct('m'), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswpunct('z'), 0);
}

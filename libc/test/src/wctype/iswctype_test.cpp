//===-- Unittests for iswctype --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/iswctype.h"
#include "src/wctype/wctype.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibciswctype, Alnum) {
  const auto desc = LIBC_NAMESPACE::wctype("alnum");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype('a', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('Z', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('5', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype('!', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(' ', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('\n', desc), 0);
}

TEST(LlvmLibciswctype, Alpha) {
  const auto desc = LIBC_NAMESPACE::wctype("alpha");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype('a', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('Z', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype('1', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(' ', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('_', desc), 0);
}

TEST(LlvmLibciswctype, Blank) {
  const auto desc = LIBC_NAMESPACE::wctype("blank");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype(' ', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('\t', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype('\n', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('\r', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('A', desc), 0);
}

TEST(LlvmLibciswctype, Cntrl) {
  const auto desc = LIBC_NAMESPACE::wctype("cntrl");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype('\0', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('\t', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('\n', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('\r', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(0x1f, desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(0x7f, desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype('A', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(' ', desc), 0);
}

TEST(LlvmLibciswctype, Digit) {
  const auto desc = LIBC_NAMESPACE::wctype("digit");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype('0', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('9', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype('a', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(' ', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('/', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(':', desc), 0);
}

TEST(LlvmLibciswctype, Graph) {
  const auto desc = LIBC_NAMESPACE::wctype("graph");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype('A', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('1', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('!', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('~', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype(' ', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('\n', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('\t', desc), 0);
}

TEST(LlvmLibciswctype, Lower) {
  const auto desc = LIBC_NAMESPACE::wctype("lower");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype('a', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('z', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype('A', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('1', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('_', desc), 0);
}

TEST(LlvmLibciswctype, Print) {
  const auto desc = LIBC_NAMESPACE::wctype("print");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype(' ', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('A', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('0', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('~', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype('\n', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('\t', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('\0', desc), 0);
}

TEST(LlvmLibciswctype, Punct) {
  const auto desc = LIBC_NAMESPACE::wctype("punct");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype('!', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('?', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('_', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('[', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype('a', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('1', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(' ', desc), 0);
}

TEST(LlvmLibciswctype, Space) {
  const auto desc = LIBC_NAMESPACE::wctype("space");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype(' ', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('\t', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('\n', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('\v', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('\f', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('\r', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype('A', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('1', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('!', desc), 0);
}

TEST(LlvmLibciswctype, Upper) {
  const auto desc = LIBC_NAMESPACE::wctype("upper");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype('A', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('Z', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype('a', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('1', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('_', desc), 0);
}

TEST(LlvmLibciswctype, XDigit) {
  const auto desc = LIBC_NAMESPACE::wctype("xdigit");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype('0', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('9', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('a', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('f', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('A', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype('F', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype('g', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('G', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('?', desc), 0);
}

TEST(LlvmLibciswctype, InvalidDescriptor) {
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('a', 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype('0', 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(' ', 0), 0);
}

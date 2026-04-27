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
  const wctype_t desc = LIBC_NAMESPACE::wctype("alnum");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'a', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'Z', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'5', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'!', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L' ', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'\n', desc), 0);
}

TEST(LlvmLibciswctype, Alpha) {
  const wctype_t desc = LIBC_NAMESPACE::wctype("alpha");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'a', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'Z', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'1', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L' ', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'_', desc), 0);
}

TEST(LlvmLibciswctype, Blank) {
  const wctype_t desc = LIBC_NAMESPACE::wctype("blank");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype(L' ', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'\t', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'\n', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'\r', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'A', desc), 0);
}

TEST(LlvmLibciswctype, Cntrl) {
  const wctype_t desc = LIBC_NAMESPACE::wctype("cntrl");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'\0', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'\t', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'\n', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'\r', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(0x1f, desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(0x7f, desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'A', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L' ', desc), 0);
}

TEST(LlvmLibciswctype, Digit) {
  const wctype_t desc = LIBC_NAMESPACE::wctype("digit");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'0', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'9', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'a', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L' ', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'/', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L':', desc), 0);
}

TEST(LlvmLibciswctype, Graph) {
  const wctype_t desc = LIBC_NAMESPACE::wctype("graph");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'A', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'1', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'!', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'~', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L' ', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'\n', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'\t', desc), 0);
}

TEST(LlvmLibciswctype, Lower) {
  const wctype_t desc = LIBC_NAMESPACE::wctype("lower");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'a', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'z', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'A', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'1', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'_', desc), 0);
}

TEST(LlvmLibciswctype, Print) {
  const wctype_t desc = LIBC_NAMESPACE::wctype("print");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype(L' ', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'A', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'0', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'~', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'\n', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'\t', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'\0', desc), 0);
}

TEST(LlvmLibciswctype, Punct) {
  const wctype_t desc = LIBC_NAMESPACE::wctype("punct");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'!', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'?', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'_', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'[', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'a', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'1', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L' ', desc), 0);
}

TEST(LlvmLibciswctype, Space) {
  const wctype_t desc = LIBC_NAMESPACE::wctype("space");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype(L' ', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'\t', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'\n', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'\v', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'\f', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'\r', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'A', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'1', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'!', desc), 0);
}

TEST(LlvmLibciswctype, Upper) {
  const wctype_t desc = LIBC_NAMESPACE::wctype("upper");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'A', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'Z', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'a', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'1', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'_', desc), 0);
}

TEST(LlvmLibciswctype, XDigit) {
  const wctype_t desc = LIBC_NAMESPACE::wctype("xdigit");
  ASSERT_NE(desc, static_cast<wctype_t>(0));

  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'0', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'9', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'a', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'f', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'A', desc), 0);
  EXPECT_NE(LIBC_NAMESPACE::iswctype(L'F', desc), 0);

  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'g', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'G', desc), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'?', desc), 0);
}

TEST(LlvmLibciswctype, InvalidDescriptor) {
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'a', 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L'0', 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::iswctype(L' ', 0), 0);
}

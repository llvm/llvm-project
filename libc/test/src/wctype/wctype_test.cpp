//===-- Unittests for wctype ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/wctype.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcwctype, ValidPropertiesReturnNonZero) {
  EXPECT_NE(LIBC_NAMESPACE::wctype("alnum"), static_cast<wctype_t>(0));
  EXPECT_NE(LIBC_NAMESPACE::wctype("alpha"), static_cast<wctype_t>(0));
  EXPECT_NE(LIBC_NAMESPACE::wctype("blank"), static_cast<wctype_t>(0));
  EXPECT_NE(LIBC_NAMESPACE::wctype("cntrl"), static_cast<wctype_t>(0));
  EXPECT_NE(LIBC_NAMESPACE::wctype("digit"), static_cast<wctype_t>(0));
  EXPECT_NE(LIBC_NAMESPACE::wctype("graph"), static_cast<wctype_t>(0));
  EXPECT_NE(LIBC_NAMESPACE::wctype("lower"), static_cast<wctype_t>(0));
  EXPECT_NE(LIBC_NAMESPACE::wctype("print"), static_cast<wctype_t>(0));
  EXPECT_NE(LIBC_NAMESPACE::wctype("punct"), static_cast<wctype_t>(0));
  EXPECT_NE(LIBC_NAMESPACE::wctype("space"), static_cast<wctype_t>(0));
  EXPECT_NE(LIBC_NAMESPACE::wctype("upper"), static_cast<wctype_t>(0));
  EXPECT_NE(LIBC_NAMESPACE::wctype("xdigit"), static_cast<wctype_t>(0));
}

TEST(LlvmLibcwctype, InvalidPropertiesReturnZero) {
  EXPECT_EQ(LIBC_NAMESPACE::wctype(nullptr), static_cast<wctype_t>(0));
  EXPECT_EQ(LIBC_NAMESPACE::wctype(""), static_cast<wctype_t>(0));
  EXPECT_EQ(LIBC_NAMESPACE::wctype("foo"), static_cast<wctype_t>(0));
  EXPECT_EQ(LIBC_NAMESPACE::wctype("Alpha"), static_cast<wctype_t>(0));
  EXPECT_EQ(LIBC_NAMESPACE::wctype("xdigit "), static_cast<wctype_t>(0));
  EXPECT_EQ(LIBC_NAMESPACE::wctype(" alnum"), static_cast<wctype_t>(0));
}

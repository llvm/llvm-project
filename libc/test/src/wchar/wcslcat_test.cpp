//===-- Unittests for wcslcat ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/wchar/wcslcat.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWCSLCatTest, TooBig) {
  const wchar_t *src = L"cd";
  wchar_t dst[4]{L"ab"};
  size_t res = LIBC_NAMESPACE::wcslcat(dst, src, 3);
  ASSERT_TRUE(dst[0] == L'a');
  ASSERT_TRUE(dst[1] == L'b');
  ASSERT_TRUE(dst[2] == L'\0');
  // Should still return src length + dst length
  ASSERT_EQ(res, size_t(4));
  // Not enough space to copy d
  res = LIBC_NAMESPACE::wcslcat(dst, src, 4);
  ASSERT_TRUE(dst[0] == L'a');
  ASSERT_TRUE(dst[1] == L'b');
  ASSERT_TRUE(dst[2] == L'c');
  ASSERT_TRUE(dst[3] == L'\0');
  ASSERT_EQ(res, size_t(4));
}

TEST(LlvmLibcWCSLCatTest, Smaller) {
  const wchar_t *src = L"cd";
  wchar_t dst[7]{L"ab"};
  size_t res = LIBC_NAMESPACE::wcslcat(dst, src, 7);
  ASSERT_TRUE(dst[0] == L'a');
  ASSERT_TRUE(dst[1] == L'b');
  ASSERT_TRUE(dst[2] == L'c');
  ASSERT_TRUE(dst[3] == L'd');
  ASSERT_TRUE(dst[4] == L'\0');
  ASSERT_EQ(res, size_t(4));
}

TEST(LlvmLibcWCSLCatTest, SmallerNoOverwriteAfter0) {
  const wchar_t *src = L"cd";
  wchar_t dst[8]{L"ab\0\0efg"};
  size_t res = LIBC_NAMESPACE::wcslcat(dst, src, 8);
  ASSERT_TRUE(dst[0] == L'a');
  ASSERT_TRUE(dst[1] == L'b');
  ASSERT_TRUE(dst[2] == L'c');
  ASSERT_TRUE(dst[3] == L'd');
  ASSERT_TRUE(dst[4] == L'\0');
  ASSERT_TRUE(dst[5] == L'f');
  ASSERT_TRUE(dst[6] == L'g');
  ASSERT_TRUE(dst[7] == L'\0');
  ASSERT_EQ(res, size_t(4));
}

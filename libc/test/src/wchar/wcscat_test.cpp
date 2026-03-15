//===-- Unittests for wcscat ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/wchar_t.h"
#include "src/wchar/wcscat.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWCSCatTest, EmptyDest) {
  // Dest should be fully replaced with src.
  wchar_t dest[4] = {L'\0'};
  const wchar_t *src = L"abc";
  LIBC_NAMESPACE::wcscat(dest, src);
  ASSERT_TRUE(dest[0] == L'a');
  ASSERT_TRUE(dest[1] == L'b');
  ASSERT_TRUE(dest[2] == L'c');
  ASSERT_TRUE(dest[3] == L'\0');
}

TEST(LlvmLibcWCSCatTest, NonEmptyDest) {
  // Src should be appended on to dest.
  wchar_t dest[7] = {L'x', L'y', L'z', L'\0'};
  const wchar_t *src = L"abc";
  LIBC_NAMESPACE::wcscat(dest, src);
  ASSERT_TRUE(dest[0] == L'x');
  ASSERT_TRUE(dest[1] == L'y');
  ASSERT_TRUE(dest[2] == L'z');
  ASSERT_TRUE(dest[3] == L'a');
  ASSERT_TRUE(dest[4] == L'b');
  ASSERT_TRUE(dest[5] == L'c');
  ASSERT_TRUE(dest[6] == L'\0');
}

TEST(LlvmLibcWCSCatTest, EmptySrc) {
  // Dest should remain intact.
  wchar_t dest[4] = {L'x', L'y', L'z', L'\0'};
  const wchar_t *src = L"";
  LIBC_NAMESPACE::wcscat(dest, src);
  ASSERT_TRUE(dest[0] == L'x');
  ASSERT_TRUE(dest[1] == L'y');
  ASSERT_TRUE(dest[2] == L'z');
  ASSERT_TRUE(dest[3] == L'\0');
}

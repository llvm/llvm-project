//===-- Unittests for wcsnlen ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/wchar/wcsnlen.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWCSNLenTest, EmptyString) {
  ASSERT_EQ(static_cast<size_t>(0), LIBC_NAMESPACE::wcsnlen(L"", 0));
  // If N is greater than string length, this should still return 0.
  ASSERT_EQ(static_cast<size_t>(0), LIBC_NAMESPACE::wcsnlen(L"", 1));
}

TEST(LlvmLibcWCSNLenTest, OneCharacterString) {
  const wchar_t *src = L"A";
  ASSERT_EQ(static_cast<size_t>(1), LIBC_NAMESPACE::wcsnlen(src, 1));
  // If N is 0, this should return 0.
  ASSERT_EQ(static_cast<size_t>(0), LIBC_NAMESPACE::wcsnlen(src, 0));
  // If N is greater than string length, this should still return 1.
  ASSERT_EQ(static_cast<size_t>(1), LIBC_NAMESPACE::wcsnlen(src, 3));
}

TEST(LlvmLibcWCSNLenTest, ManyCharacterString) {
  const wchar_t *src = L"123456789";
  ASSERT_EQ(static_cast<size_t>(9), LIBC_NAMESPACE::wcsnlen(src, 9));
  // If N is 0, this should return 0.
  ASSERT_EQ(static_cast<size_t>(0), LIBC_NAMESPACE::wcsnlen(src, 0));
  // If N is smaller than the string length, it should return N.
  ASSERT_EQ(static_cast<size_t>(3), LIBC_NAMESPACE::wcsnlen(src, 3));
  // If N is greater than string length, this should still return 9.
  ASSERT_EQ(static_cast<size_t>(9), LIBC_NAMESPACE::wcsnlen(src, 42));
}

TEST(LlvmLibcWCSNLenTest, IgnoreCharactersAfterNullTerminator) {
  const wchar_t src[5] = {L'a', L'b', L'c', L'\0', L'd'};
  ASSERT_EQ(static_cast<size_t>(3), LIBC_NAMESPACE::wcsnlen(src, 3));
  // This should only read up to the null terminator.
  ASSERT_EQ(static_cast<size_t>(3), LIBC_NAMESPACE::wcsnlen(src, 4));
  ASSERT_EQ(static_cast<size_t>(3), LIBC_NAMESPACE::wcsnlen(src, 5));
}

TEST(LlvmLibcWCSNLenTest, NoNullTerminator) {
  const wchar_t src[4] = {L'a', L'b', L'c', L'd'};
  // Should return 4
  ASSERT_EQ(static_cast<size_t>(4), LIBC_NAMESPACE::wcsnlen(src, 4));
  // Should return 2 since N is smaller than string length
  ASSERT_EQ(static_cast<size_t>(2), LIBC_NAMESPACE::wcsnlen(src, 2));
}

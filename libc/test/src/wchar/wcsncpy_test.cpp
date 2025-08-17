//===-- Unittests for wcsncpy ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/wchar_t.h"
#include "src/wchar/wcsncpy.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWCSNCpyTest, CopyZero) {
  // Dest should remain unchanged.
  wchar_t dest[3] = {L'a', L'b', L'\0'};
  const wchar_t *src = L"x";
  LIBC_NAMESPACE::wcsncpy(dest, src, 0);
  ASSERT_TRUE(dest[0] == L'a');
  ASSERT_TRUE(dest[1] == L'b');
  ASSERT_TRUE(dest[2] == L'\0');
}

TEST(LlvmLibcWCSNCpyTest, CopyFullIntoEmpty) {
  // Dest should be the exact same as src.
  wchar_t dest[15];
  const wchar_t *src = L"aaaaabbbbccccc";
  LIBC_NAMESPACE::wcsncpy(dest, src, 15);
  for (int i = 0; i < 15; i++)
    ASSERT_TRUE(dest[i] == src[i]);
}

TEST(LlvmLibcWCSNCpyTest, CopyPartial) {
  // First two characters of dest should be the first two characters of src.
  wchar_t dest[] = {L'a', L'b', L'c', L'd', L'\0'};
  const wchar_t *src = L"1234";
  LIBC_NAMESPACE::wcsncpy(dest, src, 2);
  ASSERT_TRUE(dest[0] == L'1');
  ASSERT_TRUE(dest[1] == L'2');
  ASSERT_TRUE(dest[2] == L'c');
  ASSERT_TRUE(dest[3] == L'd');
  ASSERT_TRUE(dest[4] == L'\0');
}

TEST(LlvmLibcWCSNCpyTest, CopyNullTerminator) {
  // Null terminator should copy into dest.
  wchar_t dest[] = {L'a', L'b', L'c', L'd', L'\0'};
  const wchar_t src[] = {L'\0', L'y'};
  LIBC_NAMESPACE::wcsncpy(dest, src, 1);
  ASSERT_TRUE(dest[0] == L'\0');
  ASSERT_TRUE(dest[1] == L'b');
  ASSERT_TRUE(dest[2] == L'c');
  ASSERT_TRUE(dest[3] == L'd');
  ASSERT_TRUE(dest[4] == L'\0');
}

TEST(LlvmLibcWCSNCpyTest, CopyPastSrc) {
  // Copying past src should fill with null terminator.
  wchar_t dest[] = {L'a', L'b', L'c', L'd', L'\0'};
  const wchar_t src[] = {L'x', L'\0'};
  LIBC_NAMESPACE::wcsncpy(dest, src, 4);
  ASSERT_TRUE(dest[0] == L'x');
  ASSERT_TRUE(dest[1] == L'\0');
  ASSERT_TRUE(dest[2] == L'\0');
  ASSERT_TRUE(dest[3] == L'\0');
  ASSERT_TRUE(dest[4] == L'\0');
}

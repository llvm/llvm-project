//===-- Unittests for wcslcpy ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/wchar/wcslcpy.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWCSLCpyTest, BiggerSource) {
  const wchar_t *src = L"abcde";
  wchar_t dst[3];
  size_t res = LIBC_NAMESPACE::wcslcpy(dst, src, 3);
  ASSERT_TRUE(dst[0] == L'a');
  ASSERT_TRUE(dst[1] == L'b');
  // Should append null terminator
  ASSERT_TRUE(dst[2] == L'\0');
  // Should still return length of src
  ASSERT_EQ(res, size_t(5));
}

TEST(LlvmLibcWCSLCpyTest, CopyZero) {
  const wchar_t *src = L"abcde";
  wchar_t dst = L'f';
  // Copying zero should not change destination
  size_t res = LIBC_NAMESPACE::wcslcpy(&dst, src, 0);
  ASSERT_TRUE(dst == L'f');
  // Should still return length of src
  ASSERT_EQ(res, size_t(5));
}

TEST(LlvmLibcWCSLCpyTest, SmallerSource) {
  const wchar_t *src = L"abc";
  wchar_t dst[7]{L"123456"};
  size_t res = LIBC_NAMESPACE::wcslcpy(dst, src, 7);
  ASSERT_TRUE(dst[0] == L'a');
  ASSERT_TRUE(dst[1] == L'b');
  ASSERT_TRUE(dst[2] == L'c');
  // Should append null terminator after copying source
  ASSERT_TRUE(dst[3] == L'\0');
  // Should not change following characters
  ASSERT_TRUE(dst[4] == L'5');
  ASSERT_TRUE(dst[5] == L'6');
  ASSERT_TRUE(dst[6] == L'\0');
  // Should still return length of src
  ASSERT_EQ(res, size_t(3));
}

TEST(LlvmLibcWCSLCpyTest, DoesNotCopyAfterNull) {
  const wchar_t src[5] = {L'a', L'b', L'\0', L'c', L'd'};
  wchar_t dst[5]{L"1234"};
  size_t res = LIBC_NAMESPACE::wcslcpy(dst, src, 5);
  ASSERT_TRUE(dst[0] == L'a');
  ASSERT_TRUE(dst[1] == L'b');
  ASSERT_TRUE(dst[2] == L'\0');
  // Should not change following characters
  ASSERT_TRUE(dst[3] == L'4');
  ASSERT_TRUE(dst[4] == L'\0');
  // Should still return length of src
  ASSERT_EQ(res, size_t(2));
}

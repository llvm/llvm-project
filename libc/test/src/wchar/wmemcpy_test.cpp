//===-- Unittests for wmemcpy ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/wchar/wmemcpy.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWMemcpyTest, CopyIntoEmpty) {
  wchar_t dest[10] = {};
  const wchar_t *src = L"abcde";
  LIBC_NAMESPACE::wmemcpy(dest, src, 6);
  ASSERT_TRUE(src[0] == dest[0]);
  ASSERT_TRUE(src[1] == dest[1]);
  ASSERT_TRUE(src[2] == dest[2]);
  ASSERT_TRUE(src[3] == dest[3]);
  ASSERT_TRUE(src[4] == dest[4]);
  ASSERT_TRUE(src[5] == dest[5]);
}

TEST(LlvmLibcWMemcpyTest, CopyFullString) {
  // After copying, strings should be the same.
  wchar_t dest[10] = {};
  const wchar_t *src = L"abcde";
  LIBC_NAMESPACE::wmemcpy(dest, src, 6);
  ASSERT_TRUE(src[0] == dest[0]);
  ASSERT_TRUE(src[1] == dest[1]);
  ASSERT_TRUE(src[2] == dest[2]);
  ASSERT_TRUE(src[3] == dest[3]);
  ASSERT_TRUE(src[4] == dest[4]);
  ASSERT_TRUE(src[5] == dest[5]);
}

TEST(LlvmLibcWMemcpyTest, CopyPartialString) {
  // After copying, only first two characters should be the same.
  wchar_t dest[10] = {};
  const wchar_t *src = L"abcde";
  LIBC_NAMESPACE::wmemcpy(dest, src, 2);
  ASSERT_TRUE(src[0] == dest[0]);
  ASSERT_TRUE(src[1] == dest[1]);
  ASSERT_TRUE(src[2] != dest[2]);
  ASSERT_TRUE(src[3] != dest[3]);
  ASSERT_TRUE(src[4] != dest[4]);
}

TEST(LlvmLibcWMemcpyTest, CopyZeroCharacters) {
  // Copying 0 characters should not change the string
  wchar_t dest[10] = {L'1', L'2', L'3', L'4', L'5', L'\0'};
  const wchar_t *src = L"abcde";
  LIBC_NAMESPACE::wmemcpy(dest, src, 0);
  ASSERT_TRUE(L'1' == dest[0]);
  ASSERT_TRUE(L'2' == dest[1]);
  ASSERT_TRUE(L'3' == dest[2]);
  ASSERT_TRUE(L'4' == dest[3]);
  ASSERT_TRUE(L'5' == dest[4]);
}

//===-- Unittests for wmempcpy --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wmempcpy.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWMempcpyTest, Simple) {
  const wchar_t *src = L"12345";
  wchar_t dest[10] = {};
  void *result = LIBC_NAMESPACE::wmempcpy(dest, src, 6);
  ASSERT_EQ(static_cast<wchar_t *>(result), dest + 6);

  ASSERT_TRUE(src[0] == dest[0]);
  ASSERT_TRUE(src[1] == dest[1]);
  ASSERT_TRUE(src[2] == dest[2]);
  ASSERT_TRUE(src[3] == dest[3]);
  ASSERT_TRUE(src[4] == dest[4]);
  ASSERT_TRUE(src[5] == dest[5]);
}

TEST(LlvmLibcWmempcpyTest, ZeroCount) {
  const wchar_t *src = L"12345";
  wchar_t dest[10];
  void *result = LIBC_NAMESPACE::wmempcpy(dest, src, 0);
  ASSERT_EQ(static_cast<wchar_t *>(result), dest);
}

TEST(LlvmLibcWMempcpyTest, BoundaryCheck) {
  const wchar_t *src = L"12345";
  wchar_t dest[4] = {};
  void *result = LIBC_NAMESPACE::wmempcpy(dest + 1, src + 1, 2);

  ASSERT_TRUE(dest[0] == 0);
  ASSERT_TRUE(dest[1] == src[1]);
  ASSERT_TRUE(dest[2] == src[2]);
  ASSERT_TRUE(dest[3] == 0);

  ASSERT_EQ(static_cast<wchar_t *>(result), dest + 3);
}

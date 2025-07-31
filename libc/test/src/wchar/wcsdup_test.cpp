//===-- Unittests for wcsdup ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/wchar_t.h"
#include "src/wchar/wcsdup.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcWcsDupTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcWcsDupTest, EmptyString) {
  const wchar_t *empty = L"";

  wchar_t *result = LIBC_NAMESPACE::wcsdup(empty);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_NE(result, static_cast<wchar_t *>(nullptr));
  ASSERT_NE(empty, const_cast<const wchar_t *>(result));
  ASSERT_TRUE(empty[0] == result[0]);
  ::free(result);
}

TEST_F(LlvmLibcWcsDupTest, AnyString) {
  const wchar_t *abc = L"abc";

  wchar_t *result = LIBC_NAMESPACE::wcsdup(abc);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_NE(result, static_cast<wchar_t *>(nullptr));
  ASSERT_NE(abc, const_cast<const wchar_t *>(result));
  ASSERT_TRUE(abc[0] == result[0]);
  ASSERT_TRUE(abc[1] == result[1]);
  ASSERT_TRUE(abc[2] == result[2]);
  ASSERT_TRUE(abc[3] == result[3]);
  ::free(result);
}

TEST_F(LlvmLibcWcsDupTest, NullPtr) {
  wchar_t *result = LIBC_NAMESPACE::wcsdup(nullptr);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_EQ(result, static_cast<wchar_t *>(nullptr));
}

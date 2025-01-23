//===-- Unittests for wcslen ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcslen.h"
#include "hdr/types/wchar_t.h"
#include "hdr/types/size_t.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWCSLenTest, EmptyString) {
  const wchar_t *empty = L"";

  size_t result = LIBC_NAMESPACE::wcslen(empty);
  ASSERT_EQ(size_t{0}, result);
}

TEST(LlvmLibcWCSLenTest, AnyString) {
  const wchar_t *any = L"Hello World!";

  size_t result = LIBC_NAMESPACE::wcslen(any);
  ASSERT_EQ(size_t{12}, result);
}

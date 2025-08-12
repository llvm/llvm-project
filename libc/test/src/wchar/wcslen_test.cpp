//===-- Unittests for wcslen ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/wchar/wcslen.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWCSLenTest, EmptyString) {
  ASSERT_EQ(size_t{0}, LIBC_NAMESPACE::wcslen(L""));
}

TEST(LlvmLibcWCSLenTest, AnyString) {
  ASSERT_EQ(size_t{12}, LIBC_NAMESPACE::wcslen(L"Hello World!"));
}

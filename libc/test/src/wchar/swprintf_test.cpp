//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for swprintf.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/wchar/swprintf.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSwprintfTest, StubReturnsMinusOne) {
  wchar_t buf[10];
  int result = LIBC_NAMESPACE::swprintf(buf, 10, L"test");
  ASSERT_EQ(result, -1);
}

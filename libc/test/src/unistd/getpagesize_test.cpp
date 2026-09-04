//===-- Unittests for getpagesize -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/getpagesize.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcGetPageSizeTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST(LlvmLibcGetPageSizeTest, GetPageSize) {
  // getpagesize doesn't modify errno
  int ret = LIBC_NAMESPACE::getpagesize();
  ASSERT_NE(ret, -1);
  ASSERT_NE(ret, 0);
  // Correct page size depends on the hardware mode, but will be a modulus of
  // 4096
  ASSERT_EQ(ret % 4096, 0);
}

//===---------------------- Unittests for errno --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/llvmlibc_errno.h"

#include "gtest/gtest.h"

TEST(ErrnoTest, Basic) {
  int test_val = 123;
  llvmlibc_errno = test_val;
  ASSERT_EQ(test_val, llvmlibc_errno);
}

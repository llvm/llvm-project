//===-- Unittests for errno -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcErrnoTest, Basic) {
  int test_val = 123;
  libc_errno = test_val;
  ASSERT_ERRNO_EQ(test_val);
}

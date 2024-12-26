//===-- Unittests for getpagesize -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/getpagesize.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcGetPageSizeTest, SmokeTest) {
  int r = LIBC_NAMESPACE::getpagesize();
  ASSERT_GT(r, 0);
  ASSERT_EQ(r % 1024, 0);
}

//===-- Unittests for _exit -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/_exit.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcUniStd, _exit) {
  EXPECT_EXITS([] { LIBC_NAMESPACE::_exit(1); }, 1);
  EXPECT_EXITS([] { LIBC_NAMESPACE::_exit(65); }, 65);
}

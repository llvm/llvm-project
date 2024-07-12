//===-- Unittests for quick_exit ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/exit.h"
#include "src/stdlib/quick_exit.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdlib, quick_exit) {
  EXPECT_EXITS([] { LIBC_NAMESPACE::quick_exit(1); }, 1);
  EXPECT_EXITS([] { LIBC_NAMESPACE::quick_exit(65); }, 65);
}

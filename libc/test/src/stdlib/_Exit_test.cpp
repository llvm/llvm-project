//===-- Unittests for _Exit -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/stdlib.h"
#include "src/stdlib/_Exit.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcStdlib, _Exit) {
  EXPECT_EXITS([] { __llvm_libc::_Exit(1); }, 1);
  EXPECT_EXITS([] { __llvm_libc::_Exit(65); }, 65);
}

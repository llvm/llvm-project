//===-- Unittests for _Exit -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/_Exit.h"
#include "src/stdlib/exit.h"
#include "test/UnitTest/Test.h"

#include <stdlib.h>

TEST(LlvmLibcStdlib, _Exit) {
  EXPECT_EXITS([] { __llvm_libc::_Exit(1); }, 1);
  EXPECT_EXITS([] { __llvm_libc::_Exit(65); }, 65);

  EXPECT_EXITS([] { __llvm_libc::exit(1); }, 1);
  EXPECT_EXITS([] { __llvm_libc::exit(65); }, 65);
}

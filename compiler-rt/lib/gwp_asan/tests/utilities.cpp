//===-- utilities.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/utilities.h"
#include "gwp_asan/tests/harness.h"

using gwp_asan::check;
using gwp_asan::checkWithErrorCode;

TEST(UtilitiesDeathTest, CheckPrintsAsExpected) {
  EXPECT_DEATH({ check(false, "Hello world"); }, "Hello world");
  check(true, "Should not crash");
  EXPECT_DEATH(
      { checkWithErrorCode(false, "Hello world", 1337); },
      "Hello world \\(Error Code: 1337\\)");
  EXPECT_DEATH(
      { checkWithErrorCode(false, "Hello world", -1337); },
      "Hello world \\(Error Code: -1337\\)");
}

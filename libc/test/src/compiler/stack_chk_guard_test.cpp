//===-- Unittests for __stack_chk_fail ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "src/compiler/__stack_chk_fail.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStackChkFail, Death) {
  EXPECT_DEATH([] { __stack_chk_fail(); }, WITH_SIGNAL(SIGABRT));
}

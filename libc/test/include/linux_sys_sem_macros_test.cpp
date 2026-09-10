//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for the values of the linux sys sem macros.
///
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include "include/llvm-libc-macros/linux/sys-sem-macros.h"

TEST(LlvmLibcLinuxSysSemMacrosTest, SemunIsLeftToTheProgram) {
  EXPECT_EQ(static_cast<long long>(_SEM_SEMUN_UNDEFINED), 1LL);
}

//===-- Unittests for assert ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/assert-macros.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcAssertTest, VersionMacro) {
  // 7.2p3 an integer constant expression with a value equivalent to 202311L.
  EXPECT_EQ(__STDC_VERSION_ASSERT_H__, 202311L);
}

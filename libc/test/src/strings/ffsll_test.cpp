//===-- Unittests for ffsll -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/strings/ffsll.h"

#include "src/__support/macros/config.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

TEST(LlvmLibcFfsllTest, SimpleFfsll) {
  ASSERT_EQ(ffsll(0LL), 0);
  ASSERT_EQ(ffsll(1LL), 1);
  ASSERT_EQ(ffsll(0xfbe71LL), 1);
  ASSERT_EQ(ffsll(0xfbe70LL), 5);
  ASSERT_EQ(ffsll(0x10LL), 5);
  ASSERT_EQ(ffsll(0x100LL), 9);
}

} // namespace LIBC_NAMESPACE_DECL

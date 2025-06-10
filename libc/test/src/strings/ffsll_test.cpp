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
  ASSERT_EQ(ffsll(0x0000000000000000LL), 0);
  ASSERT_EQ(ffsll(0x0000000000000001LL), 1);
  ASSERT_EQ(ffsll(0x0000000000000020LL), 6);
  ASSERT_EQ(ffsll(0x0000000000000400LL), 11);
  ASSERT_EQ(ffsll(0x0000000000008000LL), 16);
  ASSERT_EQ(ffsll(0x0000000000010000LL), 17);
  ASSERT_EQ(ffsll(0x0000000000200000LL), 22);
  ASSERT_EQ(ffsll(0x0000000004000000LL), 27);
  ASSERT_EQ(ffsll(0x0000000080000000LL), 32);
  ASSERT_EQ(ffsll(0x0000000100000000LL), 33);
  ASSERT_EQ(ffsll(0x0000002000000000LL), 38);
  ASSERT_EQ(ffsll(0x0000040000000000LL), 43);
  ASSERT_EQ(ffsll(0x0000800000000000LL), 48);
  ASSERT_EQ(ffsll(0x0001000000000000LL), 49);
  ASSERT_EQ(ffsll(0x0020000000000000LL), 54);
  ASSERT_EQ(ffsll(0x0400000000000000LL), 59);
  ASSERT_EQ(ffsll(0x8000000000000000LL), 64);
  ASSERT_EQ(ffsll(0xfbe71LL), 1);
  ASSERT_EQ(ffsll(0xfbe70LL), 5);
  ASSERT_EQ(ffsll(0x10LL), 5);
  ASSERT_EQ(ffsll(0x100LL), 9);
}

} // namespace LIBC_NAMESPACE_DECL

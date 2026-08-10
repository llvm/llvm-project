//===-- Unittests for ffsl ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/strings/ffsl.h"

#include "src/__support/macros/config.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

TEST(LlvmLibcFfslTest, SimpleFfsl) {
  ASSERT_EQ(ffsl(0x00000000L), 0);
  ASSERT_EQ(ffsl(0x00000001L), 1);
  ASSERT_EQ(ffsl(0x00000020L), 6);
  ASSERT_EQ(ffsl(0x00000400L), 11);
  ASSERT_EQ(ffsl(0x00008000L), 16);
  ASSERT_EQ(ffsl(0x00010000L), 17);
  ASSERT_EQ(ffsl(0x00200000L), 22);
  ASSERT_EQ(ffsl(0x04000000L), 27);
  ASSERT_EQ(ffsl(0x80000000L), 32);
#ifdef __LP64__
  ASSERT_EQ(ffsl(0x0000000100000000L), 33);
  ASSERT_EQ(ffsl(0x0000002000000000L), 38);
  ASSERT_EQ(ffsl(0x0000040000000000L), 43);
  ASSERT_EQ(ffsl(0x0000800000000000L), 48);
  ASSERT_EQ(ffsl(0x0001000000000000L), 49);
  ASSERT_EQ(ffsl(0x0020000000000000L), 54);
  ASSERT_EQ(ffsl(0x0400000000000000L), 59);
  ASSERT_EQ(ffsl(0x8000000000000000L), 64);
#endif
  ASSERT_EQ(ffsl(0xfbe71L), 1);
  ASSERT_EQ(ffsl(0xfbe70L), 5);
  ASSERT_EQ(ffsl(0x10L), 5);
  ASSERT_EQ(ffsl(0x100L), 9);
}

} // namespace LIBC_NAMESPACE_DECL

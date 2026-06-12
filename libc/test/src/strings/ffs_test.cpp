//===-- Unittests for ffs -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/strings/ffs.h"

#include "src/__support/macros/config.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

TEST(LlvmLibcFfsTest, SimpleFfs) {
  ASSERT_EQ(ffs(0x00000000), 0);
  ASSERT_EQ(ffs(0x00000001), 1);
  ASSERT_EQ(ffs(0x00000020), 6);
  ASSERT_EQ(ffs(0x00000400), 11);
  ASSERT_EQ(ffs(0x00008000), 16);
  ASSERT_EQ(ffs(0x00010000), 17);
  ASSERT_EQ(ffs(0x00200000), 22);
  ASSERT_EQ(ffs(0x04000000), 27);
  ASSERT_EQ(ffs(0x80000000), 32);
  ASSERT_EQ(ffs(0xfbe71), 1);
  ASSERT_EQ(ffs(0xfbe70), 5);
  ASSERT_EQ(ffs(0x10), 5);
  ASSERT_EQ(ffs(0x100), 9);
}

} // namespace LIBC_NAMESPACE_DECL

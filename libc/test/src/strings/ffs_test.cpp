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
  ASSERT_EQ(ffs(0), 0);
  ASSERT_EQ(ffs(1), 1);
  ASSERT_EQ(ffs(0xfbe71), 1);
  ASSERT_EQ(ffs(0xfbe70), 5);
  ASSERT_EQ(ffs(0x10), 5);
  ASSERT_EQ(ffs(0x100), 9);
}

} // namespace LIBC_NAMESPACE_DECL

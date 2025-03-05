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
  ASSERT_EQ(ffsl(0L), 0);
  ASSERT_EQ(ffsl(1L), 1);
  ASSERT_EQ(ffsl(0xfbe71L), 1);
  ASSERT_EQ(ffsl(0xfbe70L), 5);
  ASSERT_EQ(ffsl(0x10L), 5);
  ASSERT_EQ(ffsl(0x100L), 9);
}

} // namespace LIBC_NAMESPACE_DECL

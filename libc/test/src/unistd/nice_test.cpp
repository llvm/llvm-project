//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for nice.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/nice.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcNiceTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcNiceTest, SucceedsWithZeroIncr) {
  int current = LIBC_NAMESPACE::nice(0);
  ASSERT_ERRNO_SUCCESS();
  EXPECT_GE(current, -20);
  EXPECT_LE(current, 19);
}

TEST_F(LlvmLibcNiceTest, ClampsToMax) {
  // Increasing nice value (lowering priority) by a large amount is always
  // allowed and clamps to the maximum nice value (19 on Linux).
  int result = LIBC_NAMESPACE::nice(100);
  ASSERT_ERRNO_SUCCESS();
  EXPECT_EQ(result, 19);

  // Subsequent nice(0) confirms the process's nice value is now at 19.
  EXPECT_EQ(LIBC_NAMESPACE::nice(0), 19);
  ASSERT_ERRNO_SUCCESS();
}

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

#include "hdr/errno_macros.h"
#include "src/unistd/geteuid.h"
#include "src/unistd/nice.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcNiceTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcNiceTest, SucceedsWithZeroIncr) {
  int current = LIBC_NAMESPACE::nice(0);
  ASSERT_ERRNO_SUCCESS();
  EXPECT_GE(current, -20);
  EXPECT_LE(current, 19);
}

TEST_F(LlvmLibcNiceTest, FailsWithoutPrivilege) {
  if (LIBC_NAMESPACE::geteuid() != 0) {
    ASSERT_THAT(LIBC_NAMESPACE::nice(-1), Fails(EPERM));
  }
}

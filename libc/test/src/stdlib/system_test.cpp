//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for system.
///
//===----------------------------------------------------------------------===//

#include "hdr/sys_wait_macros.h"
#include "src/stdlib/system.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcSystemTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSystemTest, NullCommand) {
  int status = LIBC_NAMESPACE::system(nullptr);
  EXPECT_NE(status, 0);
}

TEST_F(LlvmLibcSystemTest, ValidCommandExitZero) {
  int status = LIBC_NAMESPACE::system("exit 0");
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(LlvmLibcSystemTest, ValidCommandExitNonZero) {
  int status = LIBC_NAMESPACE::system("exit 42");
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 42);
}

TEST_F(LlvmLibcSystemTest, EmptyCommand) {
  int status = LIBC_NAMESPACE::system("");
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(LlvmLibcSystemTest, NonExistentCommand) {
  int status = LIBC_NAMESPACE::system("definitely_nonexistent_command_xyz123");
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 127);
}

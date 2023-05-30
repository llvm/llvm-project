//===-- Unittest for creat ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/fcntl/creat.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcCreatTest, CreatAndOpen) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE = "testdata/creat.test";
  int fd = __llvm_libc::creat(TEST_FILE, S_IRWXU);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_GT(fd, 0);
  ASSERT_THAT(__llvm_libc::close(fd), Succeeds(0));

  fd = __llvm_libc::open(TEST_FILE, O_RDONLY);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_GT(fd, 0);
  ASSERT_THAT(__llvm_libc::close(fd), Succeeds(0));

  // TODO: 'remove' the test file at the end.
}

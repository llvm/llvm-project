//===-- Unittests for read and write --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/fsync.h"
#include "src/unistd/read.h"
#include "src/unistd/write.h"
#include "test/ErrnoSetterMatcher.h"
#include "utils/UnitTest/Test.h"
#include "utils/testutils/FDReader.h"

#include <errno.h>

TEST(LlvmLibcUniStd, WriteAndReadBackTest) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE = "__unistd_read_write.test";
  int write_fd = __llvm_libc::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_EQ(errno, 0);
  ASSERT_GT(write_fd, 0);
  constexpr const char HELLO[] = "hello";
  constexpr int HELLO_SIZE = sizeof(HELLO);
  ASSERT_THAT(__llvm_libc::write(write_fd, HELLO, HELLO_SIZE),
              Succeeds(HELLO_SIZE));
  ASSERT_THAT(__llvm_libc::fsync(write_fd), Succeeds(0));
  ASSERT_THAT(__llvm_libc::close(write_fd), Succeeds(0));

  int read_fd = __llvm_libc::open(TEST_FILE, O_RDONLY);
  ASSERT_EQ(errno, 0);
  ASSERT_GT(read_fd, 0);
  char read_buf[10];
  ASSERT_THAT(__llvm_libc::read(read_fd, read_buf, HELLO_SIZE),
              Succeeds(HELLO_SIZE));
  EXPECT_STREQ(read_buf, HELLO);
  ASSERT_THAT(__llvm_libc::close(read_fd), Succeeds(0));

  // TODO: 'remove' the test file after the test.
}

TEST(LlvmLibcUniStd, WriteFails) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;

  EXPECT_THAT(__llvm_libc::write(-1, "", 1), Fails(EBADF));
  EXPECT_THAT(__llvm_libc::write(1, reinterpret_cast<const void *>(-1), 1),
              Fails(EFAULT));
}

TEST(LlvmLibcUniStd, ReadFails) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;

  EXPECT_THAT(__llvm_libc::read(-1, nullptr, 1), Fails(EBADF));
  EXPECT_THAT(__llvm_libc::read(0, reinterpret_cast<void *>(-1), 1),
              Fails(EFAULT));
}

//===-- Unittests for dup -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/dup.h"
#include "src/unistd/read.h"
#include "src/unistd/unlink.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcdupTest, ReadAndWriteViaDup) {
  libc_errno = 0;
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE = "testdata/dup.test";
  int fd = __llvm_libc::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_GT(fd, 0);
  int dupfd = __llvm_libc::dup(fd);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_GT(dupfd, 0);

  // Write something via the dup
  constexpr char WRITE_DATA[] = "Hello, dup!";
  constexpr size_t WRITE_SIZE = sizeof(WRITE_DATA);
  ASSERT_EQ(ssize_t(WRITE_SIZE),
            __llvm_libc::write(dupfd, WRITE_DATA, WRITE_SIZE));
  ASSERT_THAT(__llvm_libc::close(dupfd), Succeeds(0));

  // Reopen the file for reading and create a dup.
  fd = __llvm_libc::open(TEST_FILE, O_RDONLY);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_GT(fd, 0);
  dupfd = __llvm_libc::dup(fd);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_GT(dupfd, 0);

  // Read the file content via the dup.
  char buf[WRITE_SIZE];
  ASSERT_THAT(__llvm_libc::read(dupfd, buf, WRITE_SIZE), Succeeds(WRITE_SIZE));
  ASSERT_STREQ(buf, WRITE_DATA);

  ASSERT_THAT(__llvm_libc::close(dupfd), Succeeds(0));
  ASSERT_THAT(__llvm_libc::unlink(TEST_FILE), Succeeds(0));
}

TEST(LlvmLibcdupTest, DupBadFD) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(__llvm_libc::dup(-1), Fails(EBADF));
}

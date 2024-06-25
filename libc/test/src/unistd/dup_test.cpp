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

#include <sys/stat.h>

TEST(LlvmLibcdupTest, ReadAndWriteViaDup) {
  LIBC_NAMESPACE::libc_errno = 0;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *FILENAME = "dup.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  int dupfd = LIBC_NAMESPACE::dup(fd);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(dupfd, 0);

  // Write something via the dup
  constexpr char WRITE_DATA[] = "Hello, dup!";
  constexpr size_t WRITE_SIZE = sizeof(WRITE_DATA);
  ASSERT_EQ(ssize_t(WRITE_SIZE),
            LIBC_NAMESPACE::write(dupfd, WRITE_DATA, WRITE_SIZE));
  ASSERT_THAT(LIBC_NAMESPACE::close(dupfd), Succeeds(0));

  // Reopen the file for reading and create a dup.
  fd = LIBC_NAMESPACE::open(TEST_FILE, O_RDONLY);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  dupfd = LIBC_NAMESPACE::dup(fd);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(dupfd, 0);

  // Read the file content via the dup.
  char buf[WRITE_SIZE];
  ASSERT_THAT(LIBC_NAMESPACE::read(dupfd, buf, WRITE_SIZE),
              Succeeds(WRITE_SIZE));
  ASSERT_STREQ(buf, WRITE_DATA);

  ASSERT_THAT(LIBC_NAMESPACE::close(dupfd), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
}

TEST(LlvmLibcdupTest, DupBadFD) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(LIBC_NAMESPACE::dup(-1), Fails(EBADF));
}

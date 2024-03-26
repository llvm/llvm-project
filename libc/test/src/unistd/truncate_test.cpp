//===-- Unittests for truncate --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/read.h"
#include "src/unistd/truncate.h"
#include "src/unistd/unlink.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/stat.h>

namespace cpp = LIBC_NAMESPACE::cpp;

TEST(LlvmLibcTruncateTest, CreateAndTruncate) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *FILENAME = "truncate.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME);
  constexpr const char WRITE_DATA[] = "hello, truncate";
  constexpr size_t WRITE_SIZE = sizeof(WRITE_DATA);
  char buf[WRITE_SIZE];

  // The test strategy is as follows:
  //   1. Create a normal file with some data in it.
  //   2. Read it to make sure what was written is actually in the file.
  //   3. Truncate to 1 byte.
  //   4. Try to read more than 1 byte and fail.
  LIBC_NAMESPACE::libc_errno = 0;
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_EQ(ssize_t(WRITE_SIZE),
            LIBC_NAMESPACE::write(fd, WRITE_DATA, WRITE_SIZE));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  fd = LIBC_NAMESPACE::open(TEST_FILE, O_RDONLY);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_EQ(ssize_t(WRITE_SIZE), LIBC_NAMESPACE::read(fd, buf, WRITE_SIZE));
  ASSERT_EQ(cpp::string_view(buf), cpp::string_view(WRITE_DATA));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::truncate(TEST_FILE, off_t(1)), Succeeds(0));

  fd = LIBC_NAMESPACE::open(TEST_FILE, O_RDONLY);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_EQ(ssize_t(1), LIBC_NAMESPACE::read(fd, buf, WRITE_SIZE));
  ASSERT_EQ(buf[0], WRITE_DATA[0]);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
}

TEST(LlvmLibcTruncateTest, TruncateNonExistentFile) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(
      LIBC_NAMESPACE::truncate("non-existent-dir/non-existent-file", off_t(1)),
      Fails(ENOENT));
}

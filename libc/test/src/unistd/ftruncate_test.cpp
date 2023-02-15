//===-- Unittests for ftruncate -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/ftruncate.h"
#include "src/unistd/read.h"
#include "src/unistd/unlink.h"
#include "src/unistd/write.h"
#include "test/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <errno.h>

namespace cpp = __llvm_libc::cpp;

TEST(LlvmLibcFtruncateTest, CreateAndTruncate) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char TEST_FILE[] = "testdata/ftruncate.test";
  constexpr const char WRITE_DATA[] = "hello, ftruncate";
  constexpr size_t WRITE_SIZE = sizeof(WRITE_DATA);
  char buf[WRITE_SIZE];

  // The test strategy is as follows:
  //   1. Create a normal file with some data in it.
  //   2. Read it to make sure what was written is actually in the file.
  //   3. Truncate to 1 byte.
  //   4. Try to read more than 1 byte and fail.
  errno = 0;
  int fd = __llvm_libc::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_EQ(errno, 0);
  ASSERT_GT(fd, 0);
  ASSERT_EQ(ssize_t(WRITE_SIZE),
            __llvm_libc::write(fd, WRITE_DATA, WRITE_SIZE));
  ASSERT_THAT(__llvm_libc::close(fd), Succeeds(0));

  fd = __llvm_libc::open(TEST_FILE, O_RDONLY);
  ASSERT_EQ(errno, 0);
  ASSERT_GT(fd, 0);
  ASSERT_EQ(ssize_t(WRITE_SIZE), __llvm_libc::read(fd, buf, WRITE_SIZE));
  ASSERT_EQ(cpp::string_view(buf), cpp::string_view(WRITE_DATA));
  ASSERT_THAT(__llvm_libc::close(fd), Succeeds(0));

  // For ftruncate operation to succeed, the file should be opened for
  // writing.
  fd = __llvm_libc::open(TEST_FILE, O_WRONLY);
  ASSERT_GT(fd, 0);
  ASSERT_EQ(errno, 0);
  ASSERT_THAT(__llvm_libc::ftruncate(fd, off_t(1)), Succeeds(0));
  ASSERT_THAT(__llvm_libc::close(fd), Succeeds(0));

  fd = __llvm_libc::open(TEST_FILE, O_RDONLY);
  ASSERT_EQ(errno, 0);
  ASSERT_GT(fd, 0);
  ASSERT_EQ(ssize_t(1), __llvm_libc::read(fd, buf, WRITE_SIZE));
  ASSERT_EQ(buf[0], WRITE_DATA[0]);
  ASSERT_THAT(__llvm_libc::close(fd), Succeeds(0));

  ASSERT_THAT(__llvm_libc::unlink(TEST_FILE), Succeeds(0));
}

TEST(LlvmLibcFtruncateTest, TruncateBadFD) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(__llvm_libc::ftruncate(1, off_t(1)), Fails(EINVAL));
}

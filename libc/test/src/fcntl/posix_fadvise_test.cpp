//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for posix_fadvise.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/fcntl/creat.h"
#include "src/fcntl/posix_fadvise.h"
#include "src/unistd/close.h"
#include "src/unistd/pipe.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcPosixFadviseTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST_F(LlvmLibcPosixFadviseTest, InvalidFileDescriptor) {
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise(-1, 0, 0, POSIX_FADV_NORMAL), EBADF);
  // posix_fadvise must return the error directly and not set errno.
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcPosixFadviseTest, ValidFile) {
  auto TEST_FILE = libc_make_test_file_path("posix_fadvise.test");
  int fd = LIBC_NAMESPACE::creat(TEST_FILE, S_IRWXU);
  ASSERT_GE(fd, 0);
  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
  });

  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise(fd, 0, 0, POSIX_FADV_NORMAL), 0);
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM), 0);
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL), 0);
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED), 0);
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED), 0);
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise(fd, 0, 0, POSIX_FADV_NOREUSE), 0);

  // Non-zero offset and length
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise(fd, 10, 20, POSIX_FADV_NORMAL), 0);

  // Invalid advice
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise(fd, 0, 0, -1), EINVAL);

  // Negative len
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise(fd, 0, -1, POSIX_FADV_NORMAL),
            EINVAL);
}

TEST_F(LlvmLibcPosixFadviseTest, Pipe) {
  int pipefd[2];
  ASSERT_THAT(LIBC_NAMESPACE::pipe(pipefd), Succeeds(0));
  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::close(pipefd[0]), Succeeds(0));
    EXPECT_THAT(LIBC_NAMESPACE::close(pipefd[1]), Succeeds(0));
  });

  // fadvise on a pipe should return ESPIPE
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise(pipefd[0], 0, 0, POSIX_FADV_NORMAL),
            ESPIPE);
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise(pipefd[1], 0, 0, POSIX_FADV_NORMAL),
            ESPIPE);
}

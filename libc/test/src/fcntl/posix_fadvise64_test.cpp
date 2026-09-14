//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for posix_fadvise64.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/off64_t.h"
#include "src/__support/CPP/scope.h"
#include "src/fcntl/creat.h"
#include "src/fcntl/posix_fadvise64.h"
#include "src/unistd/close.h"
#include "src/unistd/pipe.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcPosixFadvise64Test = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcPosixFadvise64Test, InvalidFileDescriptor) {
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise64(-1, 0, 0, POSIX_FADV_NORMAL),
            EBADF);
  // posix_fadvise64 must return the error directly and not set errno.
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcPosixFadvise64Test, ValidFile) {
  constexpr const char *TEST_FILE = "testdata/posix_fadvise64.test";
  int fd = LIBC_NAMESPACE::creat(TEST_FILE, S_IRWXU);
  ASSERT_GT(fd, 0);
  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    EXPECT_EQ(LIBC_NAMESPACE::close(fd), 0);
    EXPECT_EQ(LIBC_NAMESPACE::unlink(TEST_FILE), 0);
  });

  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise64(fd, 0, 0, POSIX_FADV_NORMAL), 0);
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise64(fd, 0, 0, POSIX_FADV_RANDOM), 0);
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise64(fd, 0, 0, POSIX_FADV_SEQUENTIAL),
            0);
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise64(fd, 0, 0, POSIX_FADV_WILLNEED), 0);
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise64(fd, 0, 0, POSIX_FADV_DONTNEED), 0);
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise64(fd, 0, 0, POSIX_FADV_NOREUSE), 0);

  // Non-zero offset and length
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise64(fd, 10, 20, POSIX_FADV_NORMAL), 0);

  // 64-bit offset and length (> 4 GiB)
  constexpr off64_t LARGE_OFFSET = static_cast<off64_t>(1) << 33;
  constexpr off64_t LARGE_LEN = static_cast<off64_t>(1) << 32;
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise64(fd, LARGE_OFFSET, LARGE_LEN,
                                            POSIX_FADV_NORMAL),
            0);

  // Invalid advice
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise64(fd, 0, 0, -1), EINVAL);

  // Negative len
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise64(fd, 0, -1, POSIX_FADV_NORMAL),
            EINVAL);
}

TEST_F(LlvmLibcPosixFadvise64Test, Pipe) {
  int pipefd[2];
  ASSERT_EQ(LIBC_NAMESPACE::pipe(pipefd), 0);
  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    EXPECT_EQ(LIBC_NAMESPACE::close(pipefd[0]), 0);
    EXPECT_EQ(LIBC_NAMESPACE::close(pipefd[1]), 0);
  });

  // fadvise on a pipe should return ESPIPE
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise64(pipefd[0], 0, 0, POSIX_FADV_NORMAL),
            ESPIPE);
  EXPECT_EQ(LIBC_NAMESPACE::posix_fadvise64(pipefd[1], 0, 0, POSIX_FADV_NORMAL),
            ESPIPE);
}

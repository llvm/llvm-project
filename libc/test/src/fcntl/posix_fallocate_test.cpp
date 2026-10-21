//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for posix_fallocate.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/off_t.h"
#include "hdr/types/struct_stat.h"
#include "src/__support/CPP/scope.h"
#include "src/fcntl/open.h"
#include "src/fcntl/posix_fallocate.h"
#include "src/sys/stat/fstat.h"
#include "src/unistd/close.h"
#include "src/unistd/pipe.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcPosixFallocateTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST_F(LlvmLibcPosixFallocateTest, InvalidArgs) {
  // Negative offset must return EINVAL.
  EXPECT_EQ(LIBC_NAMESPACE::posix_fallocate(0, -1, 1024), EINVAL);
  // Zero length may return EINVAL.
  EXPECT_EQ(LIBC_NAMESPACE::posix_fallocate(0, 0, 0), EINVAL);
  // Negative length must return EINVAL.
  EXPECT_EQ(LIBC_NAMESPACE::posix_fallocate(0, 0, -1), EINVAL);
  // posix_fallocate must return error directly and not set errno.
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcPosixFallocateTest, BadFileDescriptor) {
  EXPECT_EQ(LIBC_NAMESPACE::posix_fallocate(-1, 0, 4096), EBADF);
}

TEST_F(LlvmLibcPosixFallocateTest, AllocateAndExtend) {
  auto TEST_FILE = libc_make_test_file_path("posix_fallocate_alloc.test");
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_RDWR | O_TRUNC, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
  });

  // Allocate 4096 bytes at offset 0.
  EXPECT_EQ(LIBC_NAMESPACE::posix_fallocate(fd, 0, 4096), 0);
  struct stat st;
  ASSERT_THAT(LIBC_NAMESPACE::fstat(fd, &st), Succeeds(0));
  ASSERT_EQ(st.st_size, static_cast<off_t>(4096));

  // Extend allocation further by 2048 bytes at offset 4096.
  EXPECT_EQ(LIBC_NAMESPACE::posix_fallocate(fd, 4096, 2048), 0);
  ASSERT_THAT(LIBC_NAMESPACE::fstat(fd, &st), Succeeds(0));
  ASSERT_EQ(st.st_size, static_cast<off_t>(6144));

  // Allocation within already allocated range should succeed without changing
  // size.
  EXPECT_EQ(LIBC_NAMESPACE::posix_fallocate(fd, 1024, 1024), 0);
  ASSERT_THAT(LIBC_NAMESPACE::fstat(fd, &st), Succeeds(0));
  ASSERT_EQ(st.st_size, static_cast<off_t>(6144));
}

TEST_F(LlvmLibcPosixFallocateTest, NonZeroOffsetStart) {
  auto TEST_FILE = libc_make_test_file_path("posix_fallocate_offset.test");
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_RDWR | O_TRUNC, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
  });

  // Pre-allocate space starting at offset 1024 with length 3072 -> size should
  // become 4096.
  EXPECT_EQ(LIBC_NAMESPACE::posix_fallocate(fd, 1024, 3072), 0);
  struct stat st;
  ASSERT_THAT(LIBC_NAMESPACE::fstat(fd, &st), Succeeds(0));
  ASSERT_EQ(st.st_size, static_cast<off_t>(4096));
}

TEST_F(LlvmLibcPosixFallocateTest, Pipe) {
  int pipefd[2];
  ASSERT_THAT(LIBC_NAMESPACE::pipe(pipefd), Succeeds(0));
  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::close(pipefd[0]), Succeeds(0));
    EXPECT_THAT(LIBC_NAMESPACE::close(pipefd[1]), Succeeds(0));
  });

  // fallocate on a pipe should fail with ESPIPE or EBADF (Linux kernel
  // behavior).
  int ret = LIBC_NAMESPACE::posix_fallocate(pipefd[1], 0, 1024);
  EXPECT_TRUE(ret == ESPIPE || ret == EBADF);
}

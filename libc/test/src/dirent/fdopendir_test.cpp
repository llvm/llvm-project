//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for the POSIX fdopendir function.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/DIR.h"
#include "hdr/types/struct_dirent.h"
#include "src/dirent/closedir.h"
#include "src/dirent/dirfd.h"
#include "src/dirent/fdopendir.h"
#include "src/dirent/readdir.h"
#include "src/fcntl/fcntl.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"

#include "src/__support/CPP/string_view.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/fcntl_macros.h"

using LlvmLibcFdopendirTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST_F(LlvmLibcFdopendirTest, SuccessCase) {
  int fd = LIBC_NAMESPACE::open("testdata", O_RDONLY | O_DIRECTORY);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();

  DIR *dir = LIBC_NAMESPACE::fdopendir(fd);
  ASSERT_TRUE(dir != nullptr);
  ASSERT_ERRNO_SUCCESS();

  struct dirent *file1 = nullptr, *file2 = nullptr;
  while (true) {
    struct dirent *d = LIBC_NAMESPACE::readdir(dir);
    if (d == nullptr)
      break;
    if (LIBC_NAMESPACE::cpp::string_view(d->d_name) == "file1.txt")
      file1 = d;
    if (LIBC_NAMESPACE::cpp::string_view(d->d_name) == "file2.txt")
      file2 = d;
  }
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(file1 != nullptr);
  ASSERT_TRUE(file2 != nullptr);

  ASSERT_EQ(LIBC_NAMESPACE::closedir(dir), 0);

  // Verify fd is closed
  int fcntl_res = LIBC_NAMESPACE::fcntl(fd, F_GETFD);
  ASSERT_EQ(fcntl_res, -1);
  ASSERT_ERRNO_EQ(EBADF);
}

TEST_F(LlvmLibcFdopendirTest, InvalidFd) {
  DIR *dir = LIBC_NAMESPACE::fdopendir(-1);
  ASSERT_TRUE(dir == nullptr);
  ASSERT_ERRNO_EQ(EBADF);
}

TEST_F(LlvmLibcFdopendirTest, ClosedFd) {
  int fd = LIBC_NAMESPACE::open("testdata", O_RDONLY | O_DIRECTORY);
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  DIR *dir = LIBC_NAMESPACE::fdopendir(fd);
  ASSERT_TRUE(dir == nullptr);
  ASSERT_ERRNO_EQ(EBADF);
}

TEST_F(LlvmLibcFdopendirTest, NotADirectory) {
  int fd = LIBC_NAMESPACE::open("testdata/file1.txt", O_RDONLY);
  ASSERT_GT(fd, 0);

  DIR *dir = LIBC_NAMESPACE::fdopendir(fd);
  ASSERT_TRUE(dir == nullptr);
  ASSERT_ERRNO_EQ(ENOTDIR);

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST_F(LlvmLibcFdopendirTest, OPathFd) {
  int fd = LIBC_NAMESPACE::open("testdata", O_PATH);
  ASSERT_GT(fd, 0);

  DIR *dir = LIBC_NAMESPACE::fdopendir(fd);
  ASSERT_TRUE(dir == nullptr);
  ASSERT_ERRNO_EQ(EBADF);

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST_F(LlvmLibcFdopendirTest, WriteOnlyFd) {
  int fd = LIBC_NAMESPACE::open("testdata/file1.txt", O_WRONLY);
  ASSERT_GT(fd, 0);

  DIR *dir = LIBC_NAMESPACE::fdopendir(fd);
  ASSERT_TRUE(dir == nullptr);
  ASSERT_TRUE(libc_errno == EBADF || libc_errno == ENOTDIR);
  libc_errno = 0;

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

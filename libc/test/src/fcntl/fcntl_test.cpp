//===-- Unittest for fcntl ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/stdio_macros.h"
#include "hdr/types/struct_flock.h"
#include "src/errno/libc_errno.h"
#include "src/fcntl/fcntl.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/getpid.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/stat.h> // For S_IRWXU

TEST(LlvmLibcFcntlTest, FcntlDupfd) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE_NAME = "testdata/fcntl_dup.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);
  int fd2, fd3;
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);

  fd2 = LIBC_NAMESPACE::fcntl(fd, F_DUPFD, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd2, 0);

  fd3 = LIBC_NAMESPACE::fcntl(fd, F_DUPFD, 10);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd3, 0);

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd2), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd3), Succeeds(0));
}

TEST(LlvmLibcFcntlTest, FcntlGetFl) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE_NAME = "testdata/fcntl_getfl.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);
  int retVal;
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);

  retVal = LIBC_NAMESPACE::fcntl(fd, F_GETFL);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(retVal, -1);

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST(LlvmLibcFcntlTest, FcntlSetFl) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE_NAME = "testdata/fcntl_setfl.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);

  int retVal;
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDWR, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);

  retVal = LIBC_NAMESPACE::fcntl(fd, F_GETFL);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(retVal, -1);

  int oldFlags = LIBC_NAMESPACE::fcntl(fd, F_GETFL, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(oldFlags, 0);

  // Add the APPEND flag;
  oldFlags |= O_APPEND;

  retVal = LIBC_NAMESPACE::fcntl(fd, F_SETFL, oldFlags);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(retVal, -1);

  // Remove the APPEND flag;
  oldFlags = -oldFlags & O_APPEND;

  retVal = LIBC_NAMESPACE::fcntl(fd, F_SETFL, oldFlags);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(retVal, -1);

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST(LlvmLibcFcntlTest, FcntlGetLkRead) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE_NAME = "testdata/fcntl_getlkread.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);

  struct flock flk, svflk;
  int retVal;
  int fd =
      LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDONLY, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);

  flk.l_type = F_RDLCK;
  flk.l_start = 0;
  flk.l_whence = SEEK_SET;
  flk.l_len = 50;

  // copy flk into svflk
  svflk = flk;

  retVal = LIBC_NAMESPACE::fcntl(fd, F_GETLK, &svflk);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(retVal, -1);
  ASSERT_NE((int)flk.l_type, F_WRLCK); // File should not be write locked.

  retVal = LIBC_NAMESPACE::fcntl(fd, F_SETLK, &svflk);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(retVal, -1);

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST(LlvmLibcFcntlTest, FcntlGetLkWrite) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE_NAME = "testdata/fcntl_getlkwrite.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);

  struct flock flk, svflk;
  int retVal;
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDWR, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);

  flk.l_type = F_WRLCK;
  flk.l_start = 0;
  flk.l_whence = SEEK_SET;
  flk.l_len = 0;

  // copy flk into svflk
  svflk = flk;

  retVal = LIBC_NAMESPACE::fcntl(fd, F_GETLK, &svflk);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(retVal, -1);
  ASSERT_NE((int)flk.l_type, F_RDLCK); // File should not be read locked.

  retVal = LIBC_NAMESPACE::fcntl(fd, F_SETLK, &svflk);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(retVal, -1);

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST(LlvmLibcFcntlTest, UseAfterClose) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE_NAME = "testdata/fcntl_use_after_close.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDWR, S_IRWXU);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
  ASSERT_EQ(-1, LIBC_NAMESPACE::fcntl(fd, F_GETFL));
  ASSERT_ERRNO_EQ(EBADF);
}

TEST(LlvmLibcFcntlTest, SetGetOwnerTest) {
  LIBC_NAMESPACE::libc_errno = 0;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  pid_t pid = LIBC_NAMESPACE::getpid();
  ASSERT_GT(pid, -1);
  constexpr const char *TEST_FILE_NAME = "testdata/fcntl_set_get_owner.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDWR, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  int ret = LIBC_NAMESPACE::fcntl(fd, F_SETOWN, pid);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(ret, -1);
  int ret2 = LIBC_NAMESPACE::fcntl(fd, F_GETOWN);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(ret2, pid);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

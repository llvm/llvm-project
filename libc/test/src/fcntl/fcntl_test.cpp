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
#include "src/fcntl/fcntl.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/getpid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/stat.h> // For S_IRWXU

using LlvmLibcFcntlTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcFcntlTest, FcntlDupfd) {
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

TEST_F(LlvmLibcFcntlTest, FcntlGetFl) {
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

TEST_F(LlvmLibcFcntlTest, FcntlSetFl) {
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

/* Tests that are common between OFD and traditional variants of fcntl locks. */
template <int GETLK_CMD, int SETLK_CMD>
class LibcFcntlCommonLockTests : public LlvmLibcFcntlTest {
public:
  void GetLkRead() {
    using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
    constexpr const char *TEST_FILE_NAME = "testdata/fcntl_getlkread.test";
    const auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);

    struct flock flk = {};
    struct flock svflk = {};
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

    retVal = LIBC_NAMESPACE::fcntl(fd, GETLK_CMD, &svflk);
    ASSERT_ERRNO_SUCCESS();
    ASSERT_GT(retVal, -1);
    ASSERT_NE((int)svflk.l_type, F_WRLCK); // File should not be write locked.

    retVal = LIBC_NAMESPACE::fcntl(fd, SETLK_CMD, &svflk);
    ASSERT_ERRNO_SUCCESS();
    ASSERT_GT(retVal, -1);

    ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
  }

  void GetLkWrite() {
    using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
    constexpr const char *TEST_FILE_NAME = "testdata/fcntl_getlkwrite.test";
    const auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);

    struct flock flk = {};
    struct flock svflk = {};
    int retVal;
    int fd =
        LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDWR, S_IRWXU);
    ASSERT_ERRNO_SUCCESS();
    ASSERT_GT(fd, 0);

    flk.l_type = F_WRLCK;
    flk.l_start = 0;
    flk.l_whence = SEEK_SET;
    flk.l_len = 0;

    // copy flk into svflk
    svflk = flk;

    retVal = LIBC_NAMESPACE::fcntl(fd, GETLK_CMD, &svflk);
    ASSERT_ERRNO_SUCCESS();
    ASSERT_GT(retVal, -1);
    ASSERT_NE((int)svflk.l_type, F_RDLCK); // File should not be read locked.

    retVal = LIBC_NAMESPACE::fcntl(fd, SETLK_CMD, &svflk);
    ASSERT_ERRNO_SUCCESS();
    ASSERT_GT(retVal, -1);

    ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
  }

  void UseAfterClose() {
    using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
    constexpr const char *TEST_FILE_NAME =
        "testdata/fcntl_use_after_close.test";
    const auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);
    int fd =
        LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDWR, S_IRWXU);
    ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

    flock flk = {};
    flk.l_type = F_RDLCK;
    flk.l_start = 0;
    flk.l_whence = SEEK_SET;
    flk.l_len = 50;
    ASSERT_EQ(-1, LIBC_NAMESPACE::fcntl(fd, GETLK_CMD, &flk));
    ASSERT_ERRNO_EQ(EBADF);
  }
};

#define COMMON_LOCK_TESTS(NAME, GETLK_CMD, SETLK_CMD)                          \
  using NAME = LibcFcntlCommonLockTests<GETLK_CMD, SETLK_CMD>;                 \
  TEST_F(NAME, GetLkRead) { GetLkRead(); }                                     \
  TEST_F(NAME, GetLkWrite) { GetLkWrite(); }                                   \
  TEST_F(NAME, UseAfterClose) { UseAfterClose(); }                             \
  static_assert(true, "Require semicolon.")

COMMON_LOCK_TESTS(LlvmLibcFcntlProcessAssociatedLockTest, F_GETLK, F_SETLK);
COMMON_LOCK_TESTS(LlvmLibcFcntlOpenFileDescriptionLockTest, F_OFD_GETLK,
                  F_OFD_SETLK);

TEST_F(LlvmLibcFcntlTest, UseAfterClose) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE_NAME = "testdata/fcntl_use_after_close.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDWR, S_IRWXU);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
  ASSERT_EQ(-1, LIBC_NAMESPACE::fcntl(fd, F_GETFL));
  ASSERT_ERRNO_EQ(EBADF);
}

TEST_F(LlvmLibcFcntlTest, SetGetOwnerTest) {
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

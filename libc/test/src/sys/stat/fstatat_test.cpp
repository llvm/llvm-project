//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for fstatat.
///
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/mode_t.h"
#include "hdr/types/struct_stat.h"
#include "src/__support/CPP/scope.h"
#include "src/fcntl/open.h"
#include "src/sys/stat/fstatat.h"
#include "src/unistd/close.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcFstatatTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcFstatatTest, StatWithAtFdcwd) {
  constexpr const char *TEST_FILE = "testdata/fstatat.test";

  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_WRONLY, S_IRWXU);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();
  LIBC_NAMESPACE::cpp::scope_exit cleanup(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0)); });
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  struct stat statbuf;
  ASSERT_THAT(LIBC_NAMESPACE::fstatat(AT_FDCWD, TEST_FILE, &statbuf, 0),
              Succeeds(0));

  ASSERT_EQ(statbuf.st_mode, static_cast<mode_t>(S_IRWXU | S_IFREG));
}

TEST_F(LlvmLibcFstatatTest, StatWithDirFd) {
  constexpr const char *TEST_DIR = "testdata";
  constexpr const char *TEST_FILE = "testdata/fstatat_dir.test";
  constexpr const char *TEST_FILE_BASENAME = "fstatat_dir.test";

  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_WRONLY, S_IRWXU);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();
  LIBC_NAMESPACE::cpp::scope_exit cleanup_file(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0)); });
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  int dirfd = LIBC_NAMESPACE::open(TEST_DIR, O_DIRECTORY);
  ASSERT_GT(dirfd, 0);
  ASSERT_ERRNO_SUCCESS();
  LIBC_NAMESPACE::cpp::scope_exit cleanup_dir(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::close(dirfd), Succeeds(0)); });

  struct stat statbuf;
  ASSERT_THAT(LIBC_NAMESPACE::fstatat(dirfd, TEST_FILE_BASENAME, &statbuf, 0),
              Succeeds(0));

  ASSERT_EQ(statbuf.st_mode, static_cast<mode_t>(S_IRWXU | S_IFREG));
}

TEST_F(LlvmLibcFstatatTest, StatEmptyPath) {
  constexpr const char *TEST_FILE = "testdata/fstatat_empty.test";

  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_WRONLY, S_IRWXU);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();
  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
  });

  struct stat statbuf;
  ASSERT_THAT(LIBC_NAMESPACE::fstatat(fd, "", &statbuf, AT_EMPTY_PATH),
              Succeeds(0));

  ASSERT_EQ(statbuf.st_mode, static_cast<mode_t>(S_IRWXU | S_IFREG));
}

TEST_F(LlvmLibcFstatatTest, NonExistentFile) {
  struct stat statbuf;
  ASSERT_THAT(
      LIBC_NAMESPACE::fstatat(AT_FDCWD, "non-existent-file", &statbuf, 0),
      Fails(ENOENT));
}

TEST_F(LlvmLibcFstatatTest, BadDirFd) {
  struct stat statbuf;
  ASSERT_THAT(LIBC_NAMESPACE::fstatat(-1, "some-file", &statbuf, 0),
              Fails(EBADF));
}

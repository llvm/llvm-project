//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for mknodat.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/dev_t.h"
#include "hdr/types/mode_t.h"
#include "hdr/types/struct_stat.h"
#include "src/__support/CPP/scope.h"
#include "src/fcntl/open.h"
#include "src/sys/stat/mknodat.h"
#include "src/sys/stat/stat.h"
#include "src/sys/stat/umask.h"
#include "src/unistd/close.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcMknodatTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMknodatTest, CreateAndRemoveRegularFileWithAtFdcwd) {
  auto TEST_FILE = libc_make_test_file_path("mknodat_reg.test");
  constexpr mode_t FILE_MODE = S_IRUSR | S_IWUSR;

  mode_t old_mask = LIBC_NAMESPACE::umask(0);
  ASSERT_THAT(
      LIBC_NAMESPACE::mknodat(AT_FDCWD, TEST_FILE, S_IFREG | FILE_MODE, 0),
      Succeeds(0));
  LIBC_NAMESPACE::umask(old_mask);

  LIBC_NAMESPACE::cpp::scope_exit cleanup(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0)); });

  struct stat statbuf;
  ASSERT_THAT(LIBC_NAMESPACE::stat(TEST_FILE, &statbuf), Succeeds(0));
  ASSERT_TRUE(S_ISREG(statbuf.st_mode));
  ASSERT_EQ(statbuf.st_mode & 07777, static_cast<mode_t>(FILE_MODE));
}

TEST_F(LlvmLibcMknodatTest, CreateAndRemoveFifoWithAtFdcwd) {
  auto TEST_FIFO = libc_make_test_file_path("mknodat_fifo.test");
  constexpr mode_t FIFO_MODE = S_IRUSR | S_IWUSR;

  mode_t old_mask = LIBC_NAMESPACE::umask(0);
  ASSERT_THAT(
      LIBC_NAMESPACE::mknodat(AT_FDCWD, TEST_FIFO, S_IFIFO | FIFO_MODE, 0),
      Succeeds(0));
  LIBC_NAMESPACE::umask(old_mask);

  LIBC_NAMESPACE::cpp::scope_exit cleanup(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FIFO), Succeeds(0)); });

  struct stat statbuf;
  ASSERT_THAT(LIBC_NAMESPACE::stat(TEST_FIFO, &statbuf), Succeeds(0));
  ASSERT_TRUE(S_ISFIFO(statbuf.st_mode));
  ASSERT_EQ(statbuf.st_mode & 07777, static_cast<mode_t>(FIFO_MODE));
}

TEST_F(LlvmLibcMknodatTest, CreateAndRemoveWithDirFd) {
  auto TEST_DIR = libc_make_test_file_path("testdata");
  constexpr const char *TEST_FILE_BASENAME = "mknodat_dir.test";
  auto TEST_FILE_PATH = libc_make_test_file_path("testdata/mknodat_dir.test");
  constexpr mode_t FILE_MODE = S_IRUSR | S_IWUSR;

  int dirfd = LIBC_NAMESPACE::open(TEST_DIR, O_DIRECTORY);
  ASSERT_GT(dirfd, 0);
  LIBC_NAMESPACE::cpp::scope_exit cleanup_dir(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::close(dirfd), Succeeds(0)); });

  mode_t old_mask = LIBC_NAMESPACE::umask(0);
  ASSERT_THAT(LIBC_NAMESPACE::mknodat(dirfd, TEST_FILE_BASENAME,
                                      S_IFREG | FILE_MODE, 0),
              Succeeds(0));
  LIBC_NAMESPACE::umask(old_mask);

  LIBC_NAMESPACE::cpp::scope_exit cleanup_file([&] {
    EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_PATH), Succeeds(0));
  });

  struct stat statbuf;
  ASSERT_THAT(LIBC_NAMESPACE::stat(TEST_FILE_PATH, &statbuf), Succeeds(0));
  ASSERT_TRUE(S_ISREG(statbuf.st_mode));
  ASSERT_EQ(statbuf.st_mode & 07777, static_cast<mode_t>(FILE_MODE));
}

TEST_F(LlvmLibcMknodatTest, BadDirFd) {
  ASSERT_THAT(LIBC_NAMESPACE::mknodat(-1, "some-file", S_IFREG | 0644, 0),
              Fails(EBADF));
}

TEST_F(LlvmLibcMknodatTest, NonExistentPath) {
  auto BAD_PATH = libc_make_test_file_path("non-existent-dir/mknodat.test");
  ASSERT_THAT(LIBC_NAMESPACE::mknodat(AT_FDCWD, BAD_PATH, S_IFREG | 0644, 0),
              Fails(ENOENT));
}

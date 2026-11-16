//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for mkfifoat.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/mode_t.h"
#include "hdr/types/struct_stat.h"
#include "src/__support/CPP/scope.h"
#include "src/fcntl/open.h"
#include "src/sys/stat/mkfifoat.h"
#include "src/sys/stat/stat.h"
#include "src/sys/stat/umask.h"
#include "src/unistd/close.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcMkfifoatTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMkfifoatTest, CreateAndRemoveWithAtFdcwd) {
  constexpr const char *TEST_FIFO = "testdata/mkfifoat.testfifo";
  constexpr mode_t FIFO_MODE = S_IRUSR | S_IWUSR;

  mode_t old_mask = LIBC_NAMESPACE::umask(0);
  ASSERT_THAT(LIBC_NAMESPACE::mkfifoat(AT_FDCWD, TEST_FIFO, FIFO_MODE),
              Succeeds(0));
  LIBC_NAMESPACE::umask(old_mask);

  LIBC_NAMESPACE::cpp::scope_exit cleanup(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FIFO), Succeeds(0)); });

  struct stat statbuf;
  ASSERT_THAT(LIBC_NAMESPACE::stat(TEST_FIFO, &statbuf), Succeeds(0));
  ASSERT_TRUE(S_ISFIFO(statbuf.st_mode));
  ASSERT_EQ(statbuf.st_mode & 07777, static_cast<mode_t>(FIFO_MODE));
}

TEST_F(LlvmLibcMkfifoatTest, CreateAndRemoveWithDirFd) {
  constexpr const char *TEST_DIR = "testdata";
  constexpr const char *TEST_FIFO_BASENAME = "mkfifoat_dir.testfifo";
  constexpr const char *TEST_FIFO_PATH = "testdata/mkfifoat_dir.testfifo";
  constexpr mode_t FIFO_MODE = S_IRUSR | S_IWUSR;

  int dirfd = LIBC_NAMESPACE::open(TEST_DIR, O_DIRECTORY);
  ASSERT_GT(dirfd, 0);
  ASSERT_ERRNO_SUCCESS();
  LIBC_NAMESPACE::cpp::scope_exit cleanup_dir(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::close(dirfd), Succeeds(0)); });

  mode_t old_mask = LIBC_NAMESPACE::umask(0);
  ASSERT_THAT(LIBC_NAMESPACE::mkfifoat(dirfd, TEST_FIFO_BASENAME, FIFO_MODE),
              Succeeds(0));
  LIBC_NAMESPACE::umask(old_mask);

  LIBC_NAMESPACE::cpp::scope_exit cleanup_fifo([&] {
    EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FIFO_PATH), Succeeds(0));
  });

  struct stat statbuf;
  ASSERT_THAT(LIBC_NAMESPACE::stat(TEST_FIFO_PATH, &statbuf), Succeeds(0));
  ASSERT_TRUE(S_ISFIFO(statbuf.st_mode));
  ASSERT_EQ(statbuf.st_mode & 07777, static_cast<mode_t>(FIFO_MODE));
}

TEST_F(LlvmLibcMkfifoatTest, BadDirFd) {
  ASSERT_THAT(LIBC_NAMESPACE::mkfifoat(-1, "some-file", S_IRUSR | S_IWUSR),
              Fails(EBADF));
}

TEST_F(LlvmLibcMkfifoatTest, NonExistentPath) {
  ASSERT_THAT(LIBC_NAMESPACE::mkfifoat(
                  AT_FDCWD, "testdata/non-existent-dir/mkfifoat.testfifo",
                  S_IRUSR | S_IWUSR),
              Fails(ENOENT));
}

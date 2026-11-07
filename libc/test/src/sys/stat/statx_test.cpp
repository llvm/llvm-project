//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for statx.
///
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/mode_t.h"
#include "hdr/types/struct_statx.h"
#include "src/__support/CPP/scope.h"
#include "src/fcntl/open.h"
#include "src/sys/stat/statx.h"
#include "src/unistd/close.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcStatxTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcStatxTest, StatxWithAtFdcwd) {
  constexpr const char *TEST_FILE = "testdata/statx.test";

  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_WRONLY, S_IRWXU);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();
  LIBC_NAMESPACE::cpp::scope_exit cleanup(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0)); });
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  struct statx statxbuf;
  ASSERT_THAT(LIBC_NAMESPACE::statx(AT_FDCWD, TEST_FILE, 0, 0, &statxbuf),
              Succeeds(0));

  ASSERT_EQ(statxbuf.stx_mode, static_cast<uint16_t>(S_IRWXU | S_IFREG));
}

TEST_F(LlvmLibcStatxTest, StatxWithDirFd) {
  constexpr const char *TEST_DIR = "testdata";
  constexpr const char *TEST_FILE = "testdata/statx_dir.test";
  constexpr const char *TEST_FILE_BASENAME = "statx_dir.test";

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

  struct statx statxbuf;
  ASSERT_THAT(LIBC_NAMESPACE::statx(dirfd, TEST_FILE_BASENAME, 0, 0, &statxbuf),
              Succeeds(0));

  ASSERT_EQ(statxbuf.stx_mode, static_cast<uint16_t>(S_IRWXU | S_IFREG));
}

TEST_F(LlvmLibcStatxTest, StatxNonExistentFile) {
  struct statx statxbuf;
  EXPECT_THAT(
      LIBC_NAMESPACE::statx(AT_FDCWD, "non_existent_file", 0, 0, &statxbuf),
      Fails<int>(ENOENT));
}

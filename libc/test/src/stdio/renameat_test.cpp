//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for renameat.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/unistd_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/stdio/renameat.h"
#include "src/unistd/access.h"
#include "src/unistd/close.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcRenameatTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcRenameatTest, CreateAndRenameFileWithAtFdcwd) {
  constexpr const char *FILENAME0 = "renameat.test.file0";
  auto TEST_FILEPATH0 = libc_make_test_file_path(FILENAME0);
  constexpr const char *FILENAME1 = "renameat.test.file1";
  auto TEST_FILEPATH1 = libc_make_test_file_path(FILENAME1);

  int fd = LIBC_NAMESPACE::open(TEST_FILEPATH0, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  LIBC_NAMESPACE::cpp::scope_exit cleanup_files([&] {
    LIBC_NAMESPACE::unlink(TEST_FILEPATH0);
    LIBC_NAMESPACE::unlink(TEST_FILEPATH1);
    LIBC_NAMESPACE::libc_errno = 0;
  });

  ASSERT_THAT(LIBC_NAMESPACE::access(TEST_FILEPATH0, F_OK), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::renameat(AT_FDCWD, TEST_FILEPATH0, AT_FDCWD,
                                       TEST_FILEPATH1),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::access(TEST_FILEPATH1, F_OK), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::access(TEST_FILEPATH0, F_OK), Fails(ENOENT));
}

TEST_F(LlvmLibcRenameatTest, CreateAndRenameWithDirFd) {
  auto TEST_DIR = libc_make_test_file_path("testdata");
  constexpr const char *BASENAME0 = "renameat_dir0.test";
  constexpr const char *BASENAME1 = "renameat_dir1.test";
  auto PATH0 = libc_make_test_file_path("testdata/renameat_dir0.test");
  auto PATH1 = libc_make_test_file_path("testdata/renameat_dir1.test");

  int dirfd = LIBC_NAMESPACE::open(TEST_DIR, O_DIRECTORY);
  ASSERT_GT(dirfd, 0);
  LIBC_NAMESPACE::cpp::scope_exit cleanup_dir(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::close(dirfd), Succeeds(0)); });

  int fd = LIBC_NAMESPACE::open(PATH0, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  LIBC_NAMESPACE::cpp::scope_exit cleanup_files([&] {
    LIBC_NAMESPACE::unlink(PATH0);
    LIBC_NAMESPACE::unlink(PATH1);
    LIBC_NAMESPACE::libc_errno = 0;
  });

  ASSERT_THAT(LIBC_NAMESPACE::access(PATH0, F_OK), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::renameat(dirfd, BASENAME0, dirfd, BASENAME1),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::access(PATH1, F_OK), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::access(PATH0, F_OK), Fails(ENOENT));
}

TEST_F(LlvmLibcRenameatTest, BadDirFd) {
  ASSERT_THAT(LIBC_NAMESPACE::renameat(-1, "some-file", -1, "other-file"),
              Fails(EBADF));
}

TEST_F(LlvmLibcRenameatTest, RenameNonExistent) {
  constexpr const char *FILENAME1 = "renameat.test.nonexistent";
  auto TEST_FILEPATH1 = libc_make_test_file_path(FILENAME1);

  ASSERT_THAT(LIBC_NAMESPACE::renameat(AT_FDCWD, "non-existent-source",
                                       AT_FDCWD, TEST_FILEPATH1),
              Fails(ENOENT));
}

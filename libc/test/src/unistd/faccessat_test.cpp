//===-- Unittests for faccessat -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string.h"
#include "src/fcntl/open.h"
#include "src/sys/stat/mkdir.h"
#include "src/unistd/close.h"
#include "src/unistd/faccessat.h"
#include "src/unistd/rmdir.h"
#include "src/unistd/symlink.h"
#include "src/unistd/unlink.h"
#include "src/unistd/unlinkat.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

namespace cpp = LIBC_NAMESPACE::cpp;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

using LlvmLibcFaccessatTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcFaccessatTest, WithAtFdcwd) {
  // Test access checks on a file with AT_FDCWD and no flags, equivalent to
  // access().
  constexpr const char *FILENAME = "faccessat_basic3.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME);

  // Check permissions on a file with full permissions
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::faccessat(AT_FDCWD, TEST_FILE, F_OK, 0),
              Succeeds(0));
  ASSERT_THAT(
      LIBC_NAMESPACE::faccessat(AT_FDCWD, TEST_FILE, X_OK | W_OK | R_OK, 0),
      Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));

  // Check permissions on a file with execute-only permission
  fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IXUSR);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::faccessat(AT_FDCWD, TEST_FILE, F_OK, 0),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::faccessat(AT_FDCWD, TEST_FILE, X_OK, 0),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::faccessat(AT_FDCWD, TEST_FILE, R_OK, 0),
              Fails(EACCES));
  ASSERT_THAT(LIBC_NAMESPACE::faccessat(AT_FDCWD, TEST_FILE, W_OK, 0),
              Fails(EACCES));
  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));

  // Check permissions on a non existent file
  ASSERT_THAT(LIBC_NAMESPACE::faccessat(AT_FDCWD, TEST_FILE, F_OK, 0),
              Fails(ENOENT));
}

TEST_F(LlvmLibcFaccessatTest, RelativePathWithDirFD) {
  // Check permissions on a file releative to dir_fd
  const cpp::string DIRNAME = "faccessat_dir3";
  const cpp::string FILENAME = "relative_file3.txt";

  auto TEST_DIR = libc_make_test_file_path(DIRNAME.data());
  ASSERT_THAT(LIBC_NAMESPACE::mkdir(TEST_DIR, S_IRWXU), Succeeds(0));

  int dir_fd = LIBC_NAMESPACE::open(TEST_DIR, O_RDONLY | O_DIRECTORY);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(dir_fd, 0);

  auto FULL_FILE_PATH =
      libc_make_test_file_path((DIRNAME + "/" + FILENAME).data());
  int fd = LIBC_NAMESPACE::open(FULL_FILE_PATH, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  ASSERT_THAT(
      LIBC_NAMESPACE::faccessat(dir_fd, FILENAME.data(), R_OK | W_OK, 0),
      Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::unlinkat(dir_fd, FILENAME.data(), 0),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(dir_fd), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::rmdir(TEST_DIR), Succeeds(0));
}

TEST_F(LlvmLibcFaccessatTest, SymlinkNoFollow) {
  // Check permissions on a symlink itself, not what it links to
  constexpr const char *TARGET = "faccessat_target2";
  constexpr const char *SYMLINK = "faccessat_link2";
  auto TEST_TARGET = libc_make_test_file_path(TARGET);
  auto TEST_SYMLINK = libc_make_test_file_path(SYMLINK);

  int fd = LIBC_NAMESPACE::open(TEST_TARGET, O_WRONLY | O_CREAT, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::symlink(TARGET, TEST_SYMLINK), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::faccessat(AT_FDCWD, TEST_SYMLINK, R_OK, 0),
              Fails(EACCES));
  ASSERT_THAT(LIBC_NAMESPACE::faccessat(AT_FDCWD, TEST_SYMLINK, F_OK,
                                        AT_SYMLINK_NOFOLLOW),
              Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_SYMLINK), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_TARGET), Succeeds(0));
}

TEST_F(LlvmLibcFaccessatTest, AtEaccess) {
  // With AT_EACCESS, faccessat checks permissions using the effective user ID,
  // but the effective and real user ID will be the same here and changing that
  // is not feasible in a test, so this is just a basic sanity check.
  constexpr const char *FILENAME = "faccessat_eaccess.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME);

  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::faccessat(AT_FDCWD, TEST_FILE, X_OK | W_OK | R_OK,
                                        AT_EACCESS),
              Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
}

TEST_F(LlvmLibcFaccessatTest, AtEmptyPath) {
  constexpr const char *FILENAME = "faccessat_atemptypath.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME);

  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);

  // Check permissions on the file referred to by fd
  ASSERT_THAT(LIBC_NAMESPACE::faccessat(fd, "", F_OK, AT_EMPTY_PATH),
              Succeeds(0));
  ASSERT_THAT(
      LIBC_NAMESPACE::faccessat(fd, "", X_OK | W_OK | R_OK, AT_EMPTY_PATH),
      Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));

  // Check permissions on the current working directory
  ASSERT_THAT(LIBC_NAMESPACE::faccessat(AT_FDCWD, "", F_OK, AT_EMPTY_PATH),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::faccessat(AT_FDCWD, "", X_OK | W_OK | R_OK,
                                        AT_EMPTY_PATH),
              Succeeds(0));
}

} // namespace
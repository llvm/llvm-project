//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for chmod.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/mode_t.h"
#include "hdr/types/struct_stat.h"
#include "src/__support/CPP/scope.h"
#include "src/fcntl/open.h"
#include "src/sys/stat/chmod.h"
#include "src/sys/stat/lstat.h"
#include "src/sys/stat/stat.h"
#include "src/unistd/close.h"
#include "src/unistd/symlink.h"
#include "src/unistd/unlink.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcChmodTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcChmodTest, ChangeAndOpen) {
  // The test file is initially writable. We open it for writing and ensure
  // that it indeed can be opened for writing. Next, we close the file and
  // make it readonly using chmod. We test that chmod actually succeeded by
  // trying to open the file for writing and failing.
  constexpr const char *TEST_FILE = "testdata/chmod.test";
  const char WRITE_DATA[] = "test data";
  constexpr ssize_t WRITE_SIZE = ssize_t(sizeof(WRITE_DATA));

  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_APPEND | O_WRONLY);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(LIBC_NAMESPACE::write(fd, WRITE_DATA, sizeof(WRITE_DATA)),
            WRITE_SIZE);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  fd = LIBC_NAMESPACE::open(TEST_FILE, O_PATH);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
  EXPECT_THAT(LIBC_NAMESPACE::chmod(TEST_FILE, S_IRUSR), Succeeds(0));

  // Opening for writing should fail.
  EXPECT_EQ(LIBC_NAMESPACE::open(TEST_FILE, O_APPEND | O_WRONLY), -1);
  ASSERT_ERRNO_FAILURE();
  // But opening for reading should succeed.
  fd = LIBC_NAMESPACE::open(TEST_FILE, O_APPEND | O_RDONLY);
  EXPECT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();

  EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
  EXPECT_THAT(LIBC_NAMESPACE::chmod(TEST_FILE, S_IRWXU), Succeeds(0));
}

TEST_F(LlvmLibcChmodTest, NonExistentFile) {
  ASSERT_THAT(LIBC_NAMESPACE::chmod("non-existent-file", S_IRUSR),
              Fails(ENOENT));
}

TEST_F(LlvmLibcChmodTest, Symlink) {
  const auto TEST_FILE = libc_make_test_file_path("chmod_symlink_target.test");
  const auto TEST_FILE_LINK = libc_make_test_file_path("chmod_symlink.test");
  const auto TEST_DANGLING_LINK =
      libc_make_test_file_path("chmod_dangling.test");

  int fd =
      LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_WRONLY | O_TRUNC, S_IRWXU);
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  LIBC_NAMESPACE::cpp::scope_exit cleanup_target(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0)); });

  // Ensure initial permissions are 0700.
  ASSERT_THAT(LIBC_NAMESPACE::chmod(TEST_FILE, S_IRWXU), Succeeds(0));

  struct stat statbuf;
  ASSERT_THAT(LIBC_NAMESPACE::stat(TEST_FILE, &statbuf), Succeeds(0));
  ASSERT_EQ(statbuf.st_mode & 0777, static_cast<mode_t>(S_IRWXU));

  // Create symlink pointing to TEST_FILE.
  ASSERT_THAT(LIBC_NAMESPACE::symlink(TEST_FILE, TEST_FILE_LINK), Succeeds(0));

  LIBC_NAMESPACE::cpp::scope_exit cleanup_link([&] {
    EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_LINK), Succeeds(0));
  });

  // chmod on the symlink should modify the target file.
  EXPECT_THAT(LIBC_NAMESPACE::chmod(TEST_FILE_LINK, S_IRUSR), Succeeds(0));

  // Check that the target file mode has been modified.
  ASSERT_THAT(LIBC_NAMESPACE::stat(TEST_FILE, &statbuf), Succeeds(0));
  EXPECT_EQ(statbuf.st_mode & 0777, static_cast<mode_t>(S_IRUSR));

  // Check stat via symlink also reflects the new mode.
  ASSERT_THAT(LIBC_NAMESPACE::stat(TEST_FILE_LINK, &statbuf), Succeeds(0));
  EXPECT_EQ(statbuf.st_mode & 0777, static_cast<mode_t>(S_IRUSR));

  // Verify that the symlink itself is still a symbolic link.
  struct stat link_statbuf;
  ASSERT_THAT(LIBC_NAMESPACE::lstat(TEST_FILE_LINK, &link_statbuf),
              Succeeds(0));
  EXPECT_TRUE(S_ISLNK(link_statbuf.st_mode));

  // A dangling symlink should fail with ENOENT.
  ASSERT_THAT(
      LIBC_NAMESPACE::symlink("non-existent-target", TEST_DANGLING_LINK),
      Succeeds(0));

  LIBC_NAMESPACE::cpp::scope_exit cleanup_dangling([&] {
    EXPECT_THAT(LIBC_NAMESPACE::unlink(TEST_DANGLING_LINK), Succeeds(0));
  });

  EXPECT_THAT(LIBC_NAMESPACE::chmod(TEST_DANGLING_LINK, S_IRUSR),
              Fails(ENOENT));
}

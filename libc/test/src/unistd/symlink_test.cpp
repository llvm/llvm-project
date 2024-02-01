//===-- Unittests for symlink ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/symlink.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/stat.h>

TEST(LlvmLibcSymlinkTest, CreateAndUnlink) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *FILENAME = "symlink.test";
  auto TEST_FILE_BASE = libc_make_test_file_path(FILENAME);
  constexpr const char *FILENAME2 = "symlink.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME2);
  constexpr const char *FILENAME3 = "symlink.test.symlink";
  auto TEST_FILE_LINK = libc_make_test_file_path(FILENAME3);

  // The test strategy is as follows:
  //   1. Create a normal file
  //   2. Create a symlink to that file.
  //   3. Open the symlink to check that the symlink was created.
  //   4. Cleanup the file and its symlink.
  libc_errno = 0;
  int write_fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(write_fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(write_fd), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::symlink(TEST_FILE_BASE, TEST_FILE_LINK),
              Succeeds(0));

  int symlink_fd = LIBC_NAMESPACE::open(TEST_FILE_LINK, O_PATH);
  ASSERT_GT(symlink_fd, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::close(symlink_fd), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_LINK), Succeeds(0));
}

TEST(LlvmLibcSymlinkTest, SymlinkInNonExistentPath) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(LIBC_NAMESPACE::symlink("non-existent-dir/non-existent-file",
                                      "non-existent-dir/bad-symlink"),
              Fails(ENOENT));
}

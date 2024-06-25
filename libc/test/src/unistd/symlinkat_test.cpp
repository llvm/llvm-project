//===-- Unittests for symlinkat -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/symlinkat.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/stat.h>

TEST(LlvmLibcSymlinkatTest, CreateAndUnlink) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *FILENAME = "testdata";
  auto TEST_DIR = libc_make_test_file_path(FILENAME);
  constexpr const char *FILENAME2 = "symlinkat.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME2);
  constexpr const char *FILENAME3 = "testdata/symlinkat.test";
  auto TEST_FILE_PATH = libc_make_test_file_path(FILENAME3);
  constexpr const char *FILENAME4 = "symlinkat.test.link";
  auto TEST_FILE_LINK = libc_make_test_file_path(FILENAME4);
  constexpr const char *FILENAME5 = "testdata/symlinkat.test.link";
  auto TEST_FILE_LINK_PATH = libc_make_test_file_path(FILENAME5);

  // The test strategy is as follows:
  //   1. Create a normal file
  //   2. Create a link to that file.
  //   3. Open the link to check that the link was created.
  //   4. Cleanup the file and its link.
  LIBC_NAMESPACE::libc_errno = 0;
  int write_fd =
      LIBC_NAMESPACE::open(TEST_FILE_PATH, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(write_fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(write_fd), Succeeds(0));

  int dir_fd = LIBC_NAMESPACE::open(TEST_DIR, O_DIRECTORY);
  ASSERT_THAT(LIBC_NAMESPACE::symlinkat(TEST_FILE, dir_fd, TEST_FILE_LINK),
              Succeeds(0));

  int link_fd = LIBC_NAMESPACE::open(TEST_FILE_LINK_PATH, O_PATH);
  ASSERT_GT(link_fd, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::close(link_fd), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::close(dir_fd), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_LINK_PATH), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_PATH), Succeeds(0));
}

TEST(LlvmLibcSymlinkatTest, SymlinkInNonExistentPath) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(LIBC_NAMESPACE::symlinkat("non-existent-dir/non-existent-file",
                                        AT_FDCWD, "non-existent-dir/bad-link"),
              Fails(ENOENT));
}

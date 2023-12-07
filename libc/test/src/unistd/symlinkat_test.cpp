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

TEST(LlvmLibcSymlinkatTest, CreateAndUnlink) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_DIR = "testdata";
  constexpr const char *TEST_FILE = "symlinkat.test";
  constexpr const char *TEST_FILE_PATH = "testdata/symlinkat.test";
  constexpr const char *TEST_FILE_LINK = "symlinkat.test.link";
  constexpr const char *TEST_FILE_LINK_PATH = "testdata/symlinkat.test.link";

  // The test strategy is as follows:
  //   1. Create a normal file
  //   2. Create a link to that file.
  //   3. Open the link to check that the link was created.
  //   4. Cleanup the file and its link.
  libc_errno = 0;
  int write_fd = __llvm_libc::open(TEST_FILE_PATH, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_GT(write_fd, 0);
  ASSERT_THAT(__llvm_libc::close(write_fd), Succeeds(0));

  int dir_fd = __llvm_libc::open(TEST_DIR, O_DIRECTORY);
  ASSERT_THAT(__llvm_libc::symlinkat(TEST_FILE, dir_fd, TEST_FILE_LINK),
              Succeeds(0));

  int link_fd = __llvm_libc::open(TEST_FILE_LINK_PATH, O_PATH);
  ASSERT_GT(link_fd, 0);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_THAT(__llvm_libc::close(link_fd), Succeeds(0));

  ASSERT_THAT(__llvm_libc::close(dir_fd), Succeeds(0));
  ASSERT_THAT(__llvm_libc::unlink(TEST_FILE_LINK_PATH), Succeeds(0));
  ASSERT_THAT(__llvm_libc::unlink(TEST_FILE_PATH), Succeeds(0));
}

TEST(LlvmLibcSymlinkatTest, SymlinkInNonExistentPath) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(__llvm_libc::symlinkat("non-existent-dir/non-existent-file",
                                     AT_FDCWD, "non-existent-dir/bad-link"),
              Fails(ENOENT));
}

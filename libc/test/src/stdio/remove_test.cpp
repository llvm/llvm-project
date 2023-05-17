//===-- Unittests for remove ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/open.h"
#include "src/stdio/remove.h"
#include "src/sys/stat/mkdirat.h"
#include "src/unistd/access.h"
#include "src/unistd/close.h"
#include "test/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include "src/errno/libc_errno.h"
#include <unistd.h>

TEST(LlvmLibcRemoveTest, CreateAndRemoveFile) {
  // The test strategy is to create a file and remove it, and also verify that
  // it was removed.
  libc_errno = 0;
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE = "testdata/remove.test.file";
  int fd = __llvm_libc::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_GT(fd, 0);
  ASSERT_THAT(__llvm_libc::close(fd), Succeeds(0));

  ASSERT_THAT(__llvm_libc::access(TEST_FILE, F_OK), Succeeds(0));
  ASSERT_THAT(__llvm_libc::remove(TEST_FILE), Succeeds(0));
  ASSERT_THAT(__llvm_libc::access(TEST_FILE, F_OK), Fails(ENOENT));
}

TEST(LlvmLibcRemoveTest, CreateAndRemoveDir) {
  // The test strategy is to create a dir and remove it, and also verify that
  // it was removed.
  libc_errno = 0;
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_DIR = "testdata/remove.test.dir";
  ASSERT_THAT(__llvm_libc::mkdirat(AT_FDCWD, TEST_DIR, S_IRWXU), Succeeds(0));

  ASSERT_THAT(__llvm_libc::access(TEST_DIR, F_OK), Succeeds(0));
  ASSERT_THAT(__llvm_libc::remove(TEST_DIR), Succeeds(0));
  ASSERT_THAT(__llvm_libc::access(TEST_DIR, F_OK), Fails(ENOENT));
}

TEST(LlvmLibcRemoveTest, RemoveNonExistent) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(__llvm_libc::remove("testdata/non-existent"), Fails(ENOENT));
}

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
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include "src/errno/libc_errno.h"
#include <unistd.h>

TEST(LlvmLibcRemoveTest, CreateAndRemoveFile) {
  // The test strategy is to create a file and remove it, and also verify that
  // it was removed.
  libc_errno = 0;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_FILE = "testdata/remove.test.file";
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::access(TEST_FILE, F_OK), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::remove(TEST_FILE), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::access(TEST_FILE, F_OK), Fails(ENOENT));
}

TEST(LlvmLibcRemoveTest, CreateAndRemoveDir) {
  // The test strategy is to create a dir and remove it, and also verify that
  // it was removed.
  libc_errno = 0;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_DIR = "testdata/remove.test.dir";
  ASSERT_THAT(LIBC_NAMESPACE::mkdirat(AT_FDCWD, TEST_DIR, S_IRWXU),
              Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::access(TEST_DIR, F_OK), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::remove(TEST_DIR), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::access(TEST_DIR, F_OK), Fails(ENOENT));
}

TEST(LlvmLibcRemoveTest, RemoveNonExistent) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(LIBC_NAMESPACE::remove("testdata/non-existent"), Fails(ENOENT));
}

//===-- Unittests for unlinkat --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/fcntl/openat.h"
#include "src/unistd/close.h"
#include "src/unistd/unlinkat.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/stat.h>

TEST(LlvmLibcUnlinkatTest, CreateAndDeleteTest) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_DIR = "testdata";
  constexpr const char *TEST_FILE = "openat.test";
  int dir_fd = LIBC_NAMESPACE::open(TEST_DIR, O_DIRECTORY);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(dir_fd, 0);
  int write_fd =
      LIBC_NAMESPACE::openat(dir_fd, TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(write_fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(write_fd), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::unlinkat(dir_fd, TEST_FILE, 0), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(dir_fd), Succeeds(0));
}

TEST(LlvmLibcUnlinkatTest, UnlinkatNonExistentFile) {
  constexpr const char *TEST_DIR = "testdata";
  int dir_fd = LIBC_NAMESPACE::open(TEST_DIR, O_DIRECTORY);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(dir_fd, 0);
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  ASSERT_THAT(LIBC_NAMESPACE::unlinkat(dir_fd, "non-existent-file", 0),
              Fails(ENOENT));
  ASSERT_THAT(LIBC_NAMESPACE::close(dir_fd), Succeeds(0));
}

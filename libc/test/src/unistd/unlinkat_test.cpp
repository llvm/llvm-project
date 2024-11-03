//===-- Unittests for unlinkat --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/open.h"
#include "src/fcntl/openat.h"
#include "src/unistd/close.h"
#include "src/unistd/unlinkat.h"
#include "test/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/testutils/FDReader.h"

#include <errno.h>

TEST(LlvmLibcUnlinkatTest, CreateAndDeleteTest) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_DIR = "testdata";
  constexpr const char *TEST_FILE = "openat.test";
  int dir_fd = __llvm_libc::open(TEST_DIR, O_DIRECTORY);
  ASSERT_EQ(errno, 0);
  ASSERT_GT(dir_fd, 0);
  int write_fd =
      __llvm_libc::openat(dir_fd, TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_EQ(errno, 0);
  ASSERT_GT(write_fd, 0);
  ASSERT_THAT(__llvm_libc::close(write_fd), Succeeds(0));
  ASSERT_THAT(__llvm_libc::unlinkat(dir_fd, TEST_FILE, 0), Succeeds(0));
  ASSERT_THAT(__llvm_libc::close(dir_fd), Succeeds(0));
}

TEST(LlvmLibcUnlinkatTest, UnlinkatNonExistentFile) {
  constexpr const char *TEST_DIR = "testdata";
  int dir_fd = __llvm_libc::open(TEST_DIR, O_DIRECTORY);
  ASSERT_EQ(errno, 0);
  ASSERT_GT(dir_fd, 0);
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  ASSERT_THAT(__llvm_libc::unlinkat(dir_fd, "non-existent-file", 0),
              Fails(ENOENT));
  ASSERT_THAT(__llvm_libc::close(dir_fd), Succeeds(0));
}

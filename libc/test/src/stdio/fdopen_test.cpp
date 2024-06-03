//===-- Unittest for fcntl ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/linux/fcntl-macros.h"
#include "src/stdio/fdopen.h"

#include "src/errno/libc_errno.h"
#include "src/fcntl/fcntl.h"
#include "src/fcntl/open.h"
#include "src/stdio/fclose.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"
#include "test/UnitTest/Test.h"

#include <sys/stat.h> // For S_IRWXU

TEST(LlvmLibcStdioFdopenTest, InvalidInput) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  LIBC_NAMESPACE::libc_errno = 0;
  constexpr const char *TEST_FILE_NAME = "testdata/fdopen_invalid_inputs.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDONLY);
  auto *fp = LIBC_NAMESPACE::fdopen(fd, "m+");
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_TRUE(nullptr == fp);
  LIBC_NAMESPACE::close(fd);
  LIBC_NAMESPACE::libc_errno = 0;
  fp = LIBC_NAMESPACE::fdopen(fd, "r");
  ASSERT_ERRNO_EQ(EBADF);
  ASSERT_TRUE(nullptr == fp);
}

TEST(LlvmLibcStdioFdopenTest, InvalidMode) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  LIBC_NAMESPACE::libc_errno = 0;
  constexpr const char *TEST_FILE_NAME = "testdata/fdopen_invid_mode.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDONLY);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  auto *fp = LIBC_NAMESPACE::fdopen(fd, "r+");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(nullptr != fp);
  ASSERT_THAT(LIBC_NAMESPACE::fclose(fp), Succeeds(0));
  // If the mode argument is invalid, then `fdopen` returns a nullptr and sets
  // the `errno` to EINVAL. In this case the `mode` param can only be "r" or
  // "r+"
  int fd2 = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDONLY);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd2, 0);
  auto *fp2 = LIBC_NAMESPACE::fdopen(fd2, "w");
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_TRUE(nullptr == fp2);
}

TEST(LlvmLibcStdioFdopenTest, WriteRead) {

}

TEST(LlvmLibcStdioFdopenTest, Append) {}

TEST(LlvmLibcStdioFdopenTest, AppendPlus) {}
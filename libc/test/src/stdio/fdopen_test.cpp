//===-- Unittest for fdopen -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fdopen.h"

#include "hdr/fcntl_macros.h"
#include "src/__support/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/stdio/fclose.h"
#include "src/stdio/fgets.h"
#include "src/stdio/fputs.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/stat.h> // For S_IRWXU

TEST(LlvmLibcStdioFdopenTest, WriteAppendRead) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  libc_errno = 0;
  constexpr const char *TEST_FILE_NAME = "testdata/write_read_append.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDWR, S_IRWXU);
  auto *fp = LIBC_NAMESPACE::fdopen(fd, "w");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(nullptr != fp);
  constexpr const char HELLO[] = "Hello";
  LIBC_NAMESPACE::fputs(HELLO, fp);
  LIBC_NAMESPACE::fclose(fp);
  ASSERT_ERRNO_SUCCESS();

  constexpr const char LLVM[] = "LLVM";
  int fd2 = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_RDWR);
  auto *fp2 = LIBC_NAMESPACE::fdopen(fd2, "a");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(nullptr != fp2);
  LIBC_NAMESPACE::fputs(LLVM, fp2);
  LIBC_NAMESPACE::fclose(fp2);
  ASSERT_ERRNO_SUCCESS();

  int fd3 = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_RDWR);
  auto *fp3 = LIBC_NAMESPACE::fdopen(fd3, "r");
  char buffer[10];
  LIBC_NAMESPACE::fgets(buffer, sizeof(buffer), fp3);
  ASSERT_STREQ("HelloLLVM", buffer);
  LIBC_NAMESPACE::fclose(fp3);
  ASSERT_ERRNO_SUCCESS();
}

TEST(LlvmLibcStdioFdopenTest, InvalidFd) {
  libc_errno = 0;
  constexpr const char *TEST_FILE_NAME = "testdata/invalid_fd.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC);
  LIBC_NAMESPACE::close(fd);
  // With `fd` already closed, `fdopen` should fail and set the `errno` to EBADF
  auto *fp = LIBC_NAMESPACE::fdopen(fd, "r");
  ASSERT_ERRNO_EQ(EBADF);
  ASSERT_TRUE(nullptr == fp);
}

TEST(LlvmLibcStdioFdopenTest, InvalidMode) {
  libc_errno = 0;
  constexpr const char *TEST_FILE_NAME = "testdata/invalid_mode.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_RDONLY, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);

  // `Mode` must be one of "r", "w" or "a"
  auto *fp = LIBC_NAMESPACE::fdopen(fd, "m+");
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_TRUE(nullptr == fp);

  // If the mode argument is invalid, then `fdopen` returns a nullptr and sets
  // the `errno` to EINVAL. In this case the `mode` param can only be "r" or
  // "r+"
  auto *fp2 = LIBC_NAMESPACE::fdopen(fd, "w");
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_TRUE(nullptr == fp2);
  libc_errno = 0;
  LIBC_NAMESPACE::close(fd);
  ASSERT_ERRNO_SUCCESS();
}

//===-- Unittests for rename ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/linux/sys-stat-macros.h"
#include "include/llvm-libc-macros/linux/unistd-macros.h"
#include "src/fcntl/open.h"
#include "src/stdio/rename.h"
#include "src/unistd/access.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcRenameTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcRenameTest, CreateAndRenameFile) {
  // The test strategy is to create a file and rename it, and also verify that
  // it was renamed.
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

  constexpr const char *FILENAME0 = APPEND_LIBC_TEST("rename.test.file0");
  auto TEST_FILEPATH0 = libc_make_test_file_path(FILENAME0);

  int fd = LIBC_NAMESPACE::open(TEST_FILEPATH0, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::access(TEST_FILEPATH0, F_OK), Succeeds(0));

  constexpr const char *FILENAME1 = APPEND_LIBC_TEST("rename.test.file1");
  auto TEST_FILEPATH1 = libc_make_test_file_path(FILENAME1);
  ASSERT_THAT(LIBC_NAMESPACE::rename(TEST_FILEPATH0, TEST_FILEPATH1),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::access(TEST_FILEPATH1, F_OK), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::access(TEST_FILEPATH0, F_OK), Fails(ENOENT));
}

TEST_F(LlvmLibcRenameTest, RenameNonExistent) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;

  constexpr const char *FILENAME1 = APPEND_LIBC_TEST("rename.test.file1");
  auto TEST_FILEPATH1 = libc_make_test_file_path(FILENAME1);

  ASSERT_THAT(LIBC_NAMESPACE::rename("non-existent", TEST_FILEPATH1),
              Fails(ENOENT));
}

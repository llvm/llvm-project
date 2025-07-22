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
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <unistd.h>

using LlvmLibcRemoveTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcRemoveTest, CreateAndRemoveFile) {
  // The test strategy is to create a file and remove it, and also verify that
  // it was removed.
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

  constexpr const char *FILENAME = APPEND_LIBC_TEST("remove.test.file");
  auto TEST_FILE = libc_make_test_file_path(FILENAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::access(TEST_FILE, F_OK), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::remove(TEST_FILE), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::access(TEST_FILE, F_OK), Fails(ENOENT));
}

TEST_F(LlvmLibcRemoveTest, CreateAndRemoveDir) {
  // The test strategy is to create a dir and remove it, and also verify that
  // it was removed.
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *FILENAME = APPEND_LIBC_TEST("remove.test.dir");
  auto TEST_DIR = libc_make_test_file_path(FILENAME);
  ASSERT_THAT(LIBC_NAMESPACE::mkdirat(AT_FDCWD, TEST_DIR, S_IRWXU),
              Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::access(TEST_DIR, F_OK), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::remove(TEST_DIR), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::access(TEST_DIR, F_OK), Fails(ENOENT));
}

TEST(LlvmLibcRemoveTest, RemoveNonExistent) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(LIBC_NAMESPACE::remove("non-existent"), Fails(ENOENT));
}

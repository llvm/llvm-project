//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for syncfs.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/syncfs.h"
#include "src/unistd/unlink.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcSyncfsTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSyncfsTest, BasicSync) {
  constexpr const char *FILENAME = "syncfs.test";
  auto test_file = libc_make_test_file_path(FILENAME);
  constexpr const char WRITE_DATA[] = "hello, syncfs";
  constexpr ssize_t WRITE_SIZE = sizeof(WRITE_DATA);

  int fd = LIBC_NAMESPACE::open(test_file, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);

  LIBC_NAMESPACE::cpp::scope_exit cleanup(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::unlink(test_file), Succeeds(0)); });

  ASSERT_THAT(LIBC_NAMESPACE::write(fd, WRITE_DATA, WRITE_SIZE),
              Succeeds(WRITE_SIZE));
  ASSERT_THAT(LIBC_NAMESPACE::syncfs(fd), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST_F(LlvmLibcSyncfsTest, BadFd) {
  ASSERT_THAT(LIBC_NAMESPACE::syncfs(-1), Fails(EBADF));
}

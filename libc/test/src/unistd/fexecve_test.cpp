//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for fexecve.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/fexecve.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcFexecveTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcFexecveTest, BadFd) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  char *const argv[] = {nullptr};
  char *const envp[] = {nullptr};
  ASSERT_THAT(LIBC_NAMESPACE::fexecve(-1, argv, envp), Fails(EBADF));
}

TEST_F(LlvmLibcFexecveTest, ClosedFd) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *FILENAME = "fexecve_closed.test";
  auto test_file = libc_make_test_file_path(FILENAME);
  int fd =
      LIBC_NAMESPACE::open(test_file, O_RDONLY | O_CREAT, S_IRUSR | S_IWUSR);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(fd, 0);
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::unlink(test_file), Succeeds(0));

  char *const argv[] = {nullptr};
  char *const envp[] = {nullptr};
  ASSERT_THAT(LIBC_NAMESPACE::fexecve(fd, argv, envp), Fails(EBADF));
}

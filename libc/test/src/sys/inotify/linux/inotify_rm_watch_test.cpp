//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for inotify_rm_watch.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sys_inotify_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/sys/inotify/inotify_add_watch.h"
#include "src/sys/inotify/inotify_init.h"
#include "src/sys/inotify/inotify_rm_watch.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcInotifyRmWatchTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcInotifyRmWatchTest, Basic) {
  auto test_dir = libc_make_test_file_path(".");
  int fd;
  ASSERT_THAT(fd = LIBC_NAMESPACE::inotify_init(),
              returns(GT(0)).with_errno(EQ(0)));
  LIBC_NAMESPACE::cpp::scope_exit close_fd(
      [&] { EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0)); });

  int wd;
  ASSERT_THAT(
      wd = LIBC_NAMESPACE::inotify_add_watch(fd, test_dir, IN_ALL_EVENTS),
      returns(GT(0)).with_errno(EQ(0)));
  ASSERT_THAT(LIBC_NAMESPACE::inotify_rm_watch(fd, wd), Succeeds(0));
}

TEST_F(LlvmLibcInotifyRmWatchTest, Fail) {
  ASSERT_THAT(LIBC_NAMESPACE::inotify_rm_watch(-1, 0), Fails(EBADF));
}

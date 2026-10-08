//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for inotify_init.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sys_resource_macros.h"
#include "hdr/types/struct_rlimit.h"
#include "src/__support/CPP/scope.h"
#include "src/sys/inotify/inotify_init.h"
#include "src/sys/resource/getrlimit.h"
#include "src/sys/resource/setrlimit.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcInotifyInitTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcInotifyInitTest, Basic) {
  int fd;
  ASSERT_THAT(fd = LIBC_NAMESPACE::inotify_init(),
              returns(GT(0)).with_errno(EQ(0)));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST_F(LlvmLibcInotifyInitTest, Fail) {
  struct rlimit orig_limits;
  ASSERT_THAT(LIBC_NAMESPACE::getrlimit(RLIMIT_NOFILE, &orig_limits),
              Succeeds(0));
  LIBC_NAMESPACE::cpp::scope_exit restore_limits([&] {
    EXPECT_THAT(LIBC_NAMESPACE::setrlimit(RLIMIT_NOFILE, &orig_limits),
                Succeeds(0));
  });

  struct rlimit zero_limits{0, orig_limits.rlim_max};
  ASSERT_THAT(LIBC_NAMESPACE::setrlimit(RLIMIT_NOFILE, &zero_limits),
              Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::inotify_init(), Fails(EMFILE));
}

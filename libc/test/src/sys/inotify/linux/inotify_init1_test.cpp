//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for inotify_init1.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sys_inotify_macros.h"
#include "src/sys/inotify/inotify_init1.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcInotifyInit1Test = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcInotifyInit1Test, Basic) {
  int fd;
  ASSERT_THAT(fd = LIBC_NAMESPACE::inotify_init1(IN_CLOEXEC),
              returns(GE(0)).with_errno(EQ(0)));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}

TEST_F(LlvmLibcInotifyInit1Test, Fail) {
  ASSERT_THAT(LIBC_NAMESPACE::inotify_init1(-1), Fails(EINVAL));
}

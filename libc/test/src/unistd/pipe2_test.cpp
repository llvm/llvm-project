//===-- Unittests for pipe2 -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/errno/libc_errno.h"
#include "src/unistd/close.h"
#include "src/unistd/pipe2.h"

#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcPipe2Test, SmokeTest) {
  int pipefd[2];
  ASSERT_THAT(LIBC_NAMESPACE::pipe2(pipefd, 0), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::close(pipefd[0]), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::close(pipefd[1]), Succeeds());
}

TEST(LlvmLibcPipe2ErrTest, SmokeTest) {
  int pipefd[2];
  ASSERT_THAT(LIBC_NAMESPACE::pipe2(pipefd, -1), Fails(EINVAL));
  ASSERT_THAT(LIBC_NAMESPACE::pipe2(nullptr, 0), Fails(EFAULT));
}

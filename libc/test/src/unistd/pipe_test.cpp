//===-- Unittests for pipe ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/unistd/close.h"
#include "src/unistd/pipe.h"

#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcPipeTest, SmokeTest) {

  int pipefd[2];

  ASSERT_THAT(LIBC_NAMESPACE::pipe(pipefd), Succeeds());

  ASSERT_THAT(LIBC_NAMESPACE::close(pipefd[0]), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::close(pipefd[1]), Succeeds());
}

// TODO: Functionality tests

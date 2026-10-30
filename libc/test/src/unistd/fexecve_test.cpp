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
#include "src/unistd/fexecve.h"
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

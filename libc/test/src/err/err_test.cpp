//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for err.
///
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "src/err/err.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcErrTest, ErrExitCode) {
  libc_errno = 0;
  EXPECT_EXITS([] { LIBC_NAMESPACE::err(1, "test err"); }, 1);
  libc_errno = 2; // ENOENT
  EXPECT_EXITS([] { LIBC_NAMESPACE::err(127, "test err %d", 42); }, 127);
}

TEST(LlvmLibcErrTest, ErrNullFormat) {
  libc_errno = 2; // ENOENT
  EXPECT_EXITS([] { LIBC_NAMESPACE::err(1, nullptr); }, 1);
}

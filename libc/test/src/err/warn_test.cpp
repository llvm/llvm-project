//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for warn.
///
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "src/err/warn.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWarnTest, WarnNoExit) {
  libc_errno = 0;
  LIBC_NAMESPACE::warn("test warn");
}

TEST(LlvmLibcWarnTest, WarnNullFormat) {
  libc_errno = 2;
  LIBC_NAMESPACE::warn(nullptr);
}

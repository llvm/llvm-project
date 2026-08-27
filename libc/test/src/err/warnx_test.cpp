//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for warnx.
///
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "src/err/warnx.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWarnxTest, WarnxNoExit) {
  libc_errno = 1;
  LIBC_NAMESPACE::warnx("test warnx");
}

TEST(LlvmLibcWarnxTest, WarnxNullFormat) { LIBC_NAMESPACE::warnx(nullptr); }

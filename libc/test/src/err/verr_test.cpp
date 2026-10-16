//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for verr.
///
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "src/err/verr.h"
#include "test/UnitTest/Test.h"

#include <stdarg.h>

namespace {
void call_verr(int eval, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  LIBC_NAMESPACE::verr(eval, fmt, args);
  va_end(args);
}
} // namespace

TEST(LlvmLibcVerrTest, VerrExitCode) {
  libc_errno = 0;
  EXPECT_EXITS([] { call_verr(1, "test verr"); }, 1);
}

TEST(LlvmLibcVerrTest, VerrNullFormat) {
  libc_errno = 2;
  EXPECT_EXITS([] { call_verr(1, nullptr); }, 1);
}

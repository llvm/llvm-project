//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for verrx.
///
//===----------------------------------------------------------------------===//

#include "src/err/verrx.h"
#include "test/UnitTest/Test.h"

#include <stdarg.h>

namespace {
void call_verrx(int eval, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  LIBC_NAMESPACE::verrx(eval, fmt, args);
  va_end(args);
}
} // namespace

TEST(LlvmLibcVerrxTest, VerrxExitCode) {
  EXPECT_EXITS([] { call_verrx(2, "test verrx"); }, 2);
}

TEST(LlvmLibcVerrxTest, VerrxNullFormat) {
  EXPECT_EXITS([] { call_verrx(2, nullptr); }, 2);
}

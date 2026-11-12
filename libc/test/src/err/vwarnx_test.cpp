//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for vwarnx.
///
//===----------------------------------------------------------------------===//

#include "src/err/vwarnx.h"
#include "test/UnitTest/Test.h"

#include <stdarg.h>

namespace {
void call_vwarnx(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  LIBC_NAMESPACE::vwarnx(fmt, args);
  va_end(args);
}
} // namespace

TEST(LlvmLibcVwarnxTest, VwarnxNoExit) { call_vwarnx("test vwarnx"); }

TEST(LlvmLibcVwarnxTest, VwarnxNullFormat) { call_vwarnx(nullptr); }

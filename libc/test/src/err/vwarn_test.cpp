//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for vwarn.
///
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "src/err/vwarn.h"
#include "test/UnitTest/Test.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE {

namespace {
void call_vwarn(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vwarn(fmt, args);
  va_end(args);
}
} // namespace

TEST(LlvmLibcVwarnTest, VwarnNoExit) {
  libc_errno = 1; // EPERM
  call_vwarn("test vwarn");
}

TEST(LlvmLibcVwarnTest, VwarnNullFormat) {
  libc_errno = 2;
  call_vwarn(nullptr);
}

} // namespace LIBC_NAMESPACE

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for err family functions.
///
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/err/err.h"
#include "src/err/errx.h"
#include "src/err/verr.h"
#include "src/err/verrx.h"
#include "src/err/vwarn.h"
#include "src/err/vwarnx.h"
#include "src/err/warn.h"
#include "src/err/warnx.h"
#include "test/UnitTest/Test.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE {

namespace {
void call_verr(int eval, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  verr(eval, fmt, args);
  va_end(args);
}

void call_verrx(int eval, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  verrx(eval, fmt, args);
  va_end(args);
}

void call_vwarn(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vwarn(fmt, args);
  va_end(args);
}

void call_vwarnx(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vwarnx(fmt, args);
  va_end(args);
}
} // namespace

TEST(LlvmLibcErrTest, ErrExitCode) {
  libc_errno = 0;
  EXPECT_EXITS([] { err(1, "test err"); }, 1);
  libc_errno = 2; // ENOENT
  EXPECT_EXITS([] { err(127, "test err %d", 42); }, 127);
}

TEST(LlvmLibcErrTest, ErrxExitCode) {
  EXPECT_EXITS([] { errx(2, "test errx"); }, 2);
}

TEST(LlvmLibcErrTest, VerrExitCode) {
  libc_errno = 0;
  EXPECT_EXITS([] { call_verr(1, "test verr"); }, 1);
}

TEST(LlvmLibcErrTest, VerrxExitCode) {
  EXPECT_EXITS([] { call_verrx(2, "test verrx"); }, 2);
}

TEST(LlvmLibcErrTest, WarnNoExit) {
  libc_errno = 0;
  warn("test warn");
  libc_errno = 1;
  warnx("test warnx");
  call_vwarn("test vwarn");
  call_vwarnx("test vwarnx");
}

} // namespace LIBC_NAMESPACE

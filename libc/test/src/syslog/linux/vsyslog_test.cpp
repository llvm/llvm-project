//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for vsyslog.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/syslog_macros.h"
#include "src/__support/libc_errno.h"
#include "src/syslog/closelog.h"
#include "src/syslog/openlog.h"
#include "src/syslog/setlogmask.h"
#include "src/syslog/vsyslog.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

#include <stdarg.h>

static void call_vsyslog(int priority, const char *format, ...) {
  va_list args;
  va_start(args, format);
  LIBC_NAMESPACE::vsyslog(priority, format, args);
  va_end(args);
}

using LlvmLibcVsyslogTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcVsyslogTest, OpenLogAndClose) {
  LIBC_NAMESPACE::openlog("llvm-libc-test", LOG_PID | LOG_NDELAY, LOG_USER);
  call_vsyslog(LOG_DEBUG, "vsyslog_test %s %d", "msg", 42);
  LIBC_NAMESPACE::closelog();
}

TEST_F(LlvmLibcVsyslogTest, WithoutOpenlog) {
  call_vsyslog(LOG_USER | LOG_DEBUG, "vsyslog_test without openlog %d", 1);
  LIBC_NAMESPACE::closelog();
}

TEST_F(LlvmLibcVsyslogTest, MaskedPriority) {
  int original = LIBC_NAMESPACE::setlogmask(LOG_MASK(LOG_EMERG));
  call_vsyslog(LOG_DEBUG, "masked %m");
  LIBC_NAMESPACE::setlogmask(original);
  LIBC_NAMESPACE::closelog();
}

TEST_F(LlvmLibcVsyslogTest, PreservesErrno) {
  libc_errno = EINVAL;
  call_vsyslog(LOG_DEBUG, "vsyslog errno: %m");
  ASSERT_EQ(static_cast<int>(libc_errno), EINVAL);
  LIBC_NAMESPACE::closelog();
  ASSERT_ERRNO_EQ(EINVAL);
}

TEST_F(LlvmLibcVsyslogTest, NullFormatCrashes) {
  EXPECT_DEATH([] { call_vsyslog(LOG_DEBUG, nullptr); }, WITH_SIGNAL(-1));
}

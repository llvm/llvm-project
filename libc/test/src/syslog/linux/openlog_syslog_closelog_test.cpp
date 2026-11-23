//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for openlog, syslog and closelog.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/syslog_macros.h"
#include "src/__support/libc_errno.h"
#include "src/syslog/closelog.h"
#include "src/syslog/openlog.h"
#include "src/syslog/setlogmask.h"
#include "src/syslog/syslog.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcOpenlogSyslogCloselogTest =
    LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcOpenlogSyslogCloselogTest, CloselogWithoutOpenlog) {
  LIBC_NAMESPACE::closelog();
}

TEST_F(LlvmLibcOpenlogSyslogCloselogTest, OpenWithNdelayLogAndClose) {
  LIBC_NAMESPACE::openlog("llvm-libc-test", LOG_PID | LOG_NDELAY, LOG_USER);
  LIBC_NAMESPACE::syslog(LOG_DEBUG, "openlog_syslog_closelog_test %d", 1);
  LIBC_NAMESPACE::closelog();
}

TEST_F(LlvmLibcOpenlogSyslogCloselogTest, SyslogWithoutOpenlog) {
  LIBC_NAMESPACE::syslog(LOG_USER | LOG_DEBUG, "openlog_syslog_closelog_test");
  LIBC_NAMESPACE::closelog();
}

TEST_F(LlvmLibcOpenlogSyslogCloselogTest, CloselogTwice) {
  LIBC_NAMESPACE::openlog("llvm-libc-test", LOG_ODELAY, LOG_USER);
  LIBC_NAMESPACE::closelog();
  LIBC_NAMESPACE::closelog();
}

TEST_F(LlvmLibcOpenlogSyslogCloselogTest, OpenlogWithNullIdentAndZeroOptions) {
  LIBC_NAMESPACE::openlog(nullptr, 0, 0);
  LIBC_NAMESPACE::syslog(LOG_INFO, "null ident %s", "ok");
  LIBC_NAMESPACE::closelog();
}

TEST_F(LlvmLibcOpenlogSyslogCloselogTest, OpenlogReconfigureAndOptions) {
  LIBC_NAMESPACE::openlog("first-ident", LOG_ODELAY | LOG_NOWAIT, LOG_LOCAL0);
  LIBC_NAMESPACE::openlog("second-ident", LOG_PID | LOG_CONS | LOG_NDELAY,
                          LOG_LOCAL1);
  LIBC_NAMESPACE::syslog(LOG_DEBUG, "reconfigured");
  LIBC_NAMESPACE::syslog(LOG_LOCAL2 | LOG_INFO, "explicit facility");
  LIBC_NAMESPACE::closelog();
}

TEST_F(LlvmLibcOpenlogSyslogCloselogTest, MaskedPriorityAndMaskPersistence) {
  int original = LIBC_NAMESPACE::setlogmask(LOG_MASK(LOG_EMERG));
  LIBC_NAMESPACE::openlog("llvm-libc-test", LOG_ODELAY, LOG_USER);
  ASSERT_EQ(LIBC_NAMESPACE::setlogmask(0), LOG_MASK(LOG_EMERG));
  LIBC_NAMESPACE::syslog(LOG_DEBUG, "masked %m");
  LIBC_NAMESPACE::syslog(LOG_LOCAL0 | LOG_DEBUG, "masked with facility");
  LIBC_NAMESPACE::closelog();
  ASSERT_EQ(LIBC_NAMESPACE::setlogmask(0), LOG_MASK(LOG_EMERG));
  LIBC_NAMESPACE::setlogmask(original);
}

TEST_F(LlvmLibcOpenlogSyslogCloselogTest, PreservesErrno) {
  libc_errno = EINVAL;
  LIBC_NAMESPACE::openlog("llvm-libc-test", LOG_PID | LOG_ODELAY, LOG_USER);
  ASSERT_EQ(static_cast<int>(libc_errno), EINVAL);
  LIBC_NAMESPACE::syslog(LOG_DEBUG, "errno preserved: %m");
  ASSERT_EQ(static_cast<int>(libc_errno), EINVAL);
  LIBC_NAMESPACE::closelog();
  ASSERT_ERRNO_EQ(EINVAL);
}

TEST_F(LlvmLibcOpenlogSyslogCloselogTest, NullFormatCrashes) {
  EXPECT_DEATH([] { LIBC_NAMESPACE::syslog(LOG_DEBUG, nullptr); },
               WITH_SIGNAL(-1));
}

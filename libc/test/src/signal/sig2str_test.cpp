//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for sig2str and str2sig.
///
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "src/signal/sig2str.h"
#include "src/signal/str2sig.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSig2StrTest, StandardSignals) {
  char buf[SIG2STR_MAX];

  EXPECT_EQ(LIBC_NAMESPACE::sig2str(SIGINT, buf), 0);
  EXPECT_STREQ(buf, "INT");

  EXPECT_EQ(LIBC_NAMESPACE::sig2str(SIGKILL, buf), 0);
  EXPECT_STREQ(buf, "KILL");

  EXPECT_EQ(LIBC_NAMESPACE::sig2str(SIGTERM, buf), 0);
  EXPECT_STREQ(buf, "TERM");

  EXPECT_EQ(LIBC_NAMESPACE::sig2str(SIGABRT, buf), 0);
  EXPECT_STREQ(buf, "ABRT");
}

TEST(LlvmLibcSig2StrTest, RealtimeSignals) {
  char buf[SIG2STR_MAX];

#if defined(SIGRTMIN) && defined(SIGRTMAX)
  EXPECT_EQ(LIBC_NAMESPACE::sig2str(SIGRTMIN, buf), 0);
  EXPECT_STREQ(buf, "RTMIN");

  EXPECT_EQ(LIBC_NAMESPACE::sig2str(SIGRTMAX, buf), 0);
  EXPECT_STREQ(buf, "RTMAX");

  if (SIGRTMIN + 1 < SIGRTMAX) {
    EXPECT_EQ(LIBC_NAMESPACE::sig2str(SIGRTMIN + 1, buf), 0);
    EXPECT_STREQ(buf, "RTMIN+1");
  }
#endif
}

TEST(LlvmLibcSig2StrTest, InvalidSignals) {
  char buf[SIG2STR_MAX];
  EXPECT_EQ(LIBC_NAMESPACE::sig2str(0, buf), -1);
  EXPECT_EQ(LIBC_NAMESPACE::sig2str(-1, buf), -1);
  EXPECT_EQ(LIBC_NAMESPACE::sig2str(12345, buf), -1);
  EXPECT_EQ(LIBC_NAMESPACE::sig2str(SIGINT, nullptr), -1);
}

TEST(LlvmLibcStr2SigTest, StandardNames) {
  int signum = -1;

  EXPECT_EQ(LIBC_NAMESPACE::str2sig("INT", &signum), 0);
  EXPECT_EQ(signum, SIGINT);

  EXPECT_EQ(LIBC_NAMESPACE::str2sig("KILL", &signum), 0);
  EXPECT_EQ(signum, SIGKILL);

  EXPECT_EQ(LIBC_NAMESPACE::str2sig("TERM", &signum), 0);
  EXPECT_EQ(signum, SIGTERM);

  EXPECT_EQ(LIBC_NAMESPACE::str2sig("ABRT", &signum), 0);
  EXPECT_EQ(signum, SIGABRT);
}

TEST(LlvmLibcStr2SigTest, RealtimeNames) {
  int signum = -1;

#if defined(SIGRTMIN) && defined(SIGRTMAX)
  EXPECT_EQ(LIBC_NAMESPACE::str2sig("RTMIN", &signum), 0);
  EXPECT_EQ(signum, SIGRTMIN);

  EXPECT_EQ(LIBC_NAMESPACE::str2sig("RTMAX", &signum), 0);
  EXPECT_EQ(signum, SIGRTMAX);

  if (SIGRTMIN + 1 < SIGRTMAX) {
    EXPECT_EQ(LIBC_NAMESPACE::str2sig("RTMIN+1", &signum), 0);
    EXPECT_EQ(signum, SIGRTMIN + 1);
  }
#endif
}

TEST(LlvmLibcStr2SigTest, NumericStrings) {
  int signum = -1;

  EXPECT_EQ(LIBC_NAMESPACE::str2sig("9", &signum), 0);
  EXPECT_EQ(signum, 9);
}

TEST(LlvmLibcStr2SigTest, InvalidStrings) {
  int signum = -1;

  EXPECT_EQ(LIBC_NAMESPACE::str2sig("SIGINT", &signum), -1);
  EXPECT_EQ(LIBC_NAMESPACE::str2sig("NONEXISTENT", &signum), -1);
  EXPECT_EQ(LIBC_NAMESPACE::str2sig("", &signum), -1);
  EXPECT_EQ(LIBC_NAMESPACE::str2sig(nullptr, &signum), -1);
  EXPECT_EQ(LIBC_NAMESPACE::str2sig("INT", nullptr), -1);
}

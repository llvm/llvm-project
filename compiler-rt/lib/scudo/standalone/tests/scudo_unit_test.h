//===-- scudo_unit_test.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "platform.h"

#if SCUDO_FUCHSIA
#include <zxtest/zxtest.h>
using Test = ::zxtest::Test;
#define TEST_SKIP(message) ZXTEST_SKIP(message)
#define TEST_HAS_FAILURE true
#else
#include "gtest/gtest.h"
using Test = ::testing::Test;
#define TEST_SKIP(message)                                                     \
  do {                                                                         \
    GTEST_SKIP() << message;                                                   \
  } while (0)
#define TEST_HAS_FAILURE Test::HasFailure()
#endif

// If EXPECT_DEATH isn't defined, make it a no-op.
#ifndef EXPECT_DEATH
// If ASSERT_DEATH is defined, make EXPECT_DEATH a wrapper to it.
#ifdef ASSERT_DEATH
#define EXPECT_DEATH(X, Y) ASSERT_DEATH(([&] { X; }), "")
#else
#define EXPECT_DEATH(X, Y)                                                     \
  do {                                                                         \
  } while (0)
#endif // ASSERT_DEATH
#endif // EXPECT_DEATH

// If EXPECT_STREQ isn't defined, define our own simple one.
#ifndef EXPECT_STREQ
#define EXPECT_STREQ(X, Y) EXPECT_EQ(strcmp(X, Y), 0)
#endif

#if SCUDO_FUCHSIA
#define SKIP_ON_FUCHSIA(T) DISABLED_##T
#else
#define SKIP_ON_FUCHSIA(T) T
#endif

#if SCUDO_DEBUG
#define SKIP_NO_DEBUG(T) T
#else
#define SKIP_NO_DEBUG(T) DISABLED_##T
#endif

#if SCUDO_FUCHSIA
// The zxtest library provides a default main function that does the same thing
// for Fuchsia builds.
#define SCUDO_NO_TEST_MAIN
#endif

#if SCUDO_ANDROID
static void DisableDebuggerdMaybe() {
  // Disable the debuggerd signal handler on Android, without this we can end
  // up spending a significant amount of time creating tombstones.
  signal(SIGSEGV, SIG_DFL);
  signal(SIGABRT, SIG_DFL);
}

#define SCUDO_EXPECT_DEATH(X, Y)                                               \
  EXPECT_DEATH(                                                                \
      {                                                                        \
        DisableDebuggerdMaybe();                                               \
        X;                                                                     \
      },                                                                       \
      Y);
#define SCUDO_ASSERT_DEATH(X, Y)                                               \
  ASSERT_DEATH(                                                                \
      {                                                                        \
        DisableDebuggerdMaybe();                                               \
        X;                                                                     \
      },                                                                       \
      Y);
#else

#define SCUDO_EXPECT_DEATH(X, Y) EXPECT_DEATH(X, Y);
#define SCUDO_ASSERT_DEATH(X, Y) ASSERT_DEATH(X, Y);

#endif

extern bool UseQuarantine;

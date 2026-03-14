///===- ThreadCrashReporterTests.cpp - Thread local signal handling tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/ThreadCrashReporter.h"
#include "support/Threading.h"
#include "llvm/Support/Signals.h"
#include "gtest/gtest.h"
#include <csignal>
#include <string>

namespace clang {
namespace clangd {

namespace {

static void infoSignalHandler() { ThreadCrashReporter::runCrashHandlers(); }

TEST(ThreadCrashReporterTest, All) {
#if defined(_WIN32)
  // Simulate signals on Windows for unit testing purposes.
  // The `crash.test` lit test checks the end-to-end integration.
  auto SignalCurrentThread = []() { infoSignalHandler(); };
#else
  llvm::sys::SetInfoSignalFunction(&infoSignalHandler);
  auto SignalCurrentThread = []() { raise(SIGUSR1); };
#endif

  AsyncTaskRunner Runner;
  auto SignalAnotherThread = [&]() {
    Runner.runAsync("signal another thread", SignalCurrentThread);
    Runner.wait();
  };

  bool Called;
  {
    ThreadCrashReporter ScopedReporter([&Called]() { Called = true; });
    // Check handler gets called when a signal gets delivered to the current
    // thread.
    Called = false;
    SignalCurrentThread();
    EXPECT_TRUE(Called);

    // Check handler does not get called when another thread gets signalled.
    Called = false;
    SignalAnotherThread();
    EXPECT_FALSE(Called);
  }
  // Check handler does not get called when the reporter object goes out of
  // scope.
  Called = false;
  SignalCurrentThread();
  EXPECT_FALSE(Called);

  std::string Order = "";
  {
    ThreadCrashReporter ScopedReporter([&Order] { Order.push_back('a'); });
    {
      ThreadCrashReporter ScopedReporter([&Order] { Order.push_back('b'); });
      SignalCurrentThread();
    }
    // Check that handlers are called in LIFO order.
    EXPECT_EQ(Order, "ba");

    // Check that current handler is the only one after the nested scope is
    // over.
    SignalCurrentThread();
    EXPECT_EQ(Order, "baa");
  }
}

} // namespace
} // namespace clangd
} // namespace clang

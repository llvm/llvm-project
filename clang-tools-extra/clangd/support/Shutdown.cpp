//===--- Shutdown.cpp - Unclean exit scenarios ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/Shutdown.h"

#include <atomic>
#include <cstdlib>
#include <thread>

namespace clang {
namespace clangd {

void abortAfterTimeout(std::chrono::seconds Timeout) {
  // This is more portable than sys::WatchDog, and yields a stack trace.
  std::thread([Timeout] {
    std::this_thread::sleep_for(Timeout);
    std::abort();
  }).detach();
}

static std::atomic<bool> ShutdownRequested = {false};

void requestShutdown() {
  if (ShutdownRequested.exchange(true))
    // This is the second shutdown request. Exit hard.
    std::abort();
}

bool shutdownRequested() { return ShutdownRequested; }

} // namespace clangd
} // namespace clang

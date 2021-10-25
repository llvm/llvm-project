//===--- ThreadCrashReporter.cpp - Thread local signal handling --*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/ThreadCrashReporter.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ThreadLocal.h"

namespace clang {
namespace clangd {

static thread_local ThreadCrashReporter *CurrentReporter = nullptr;

void ThreadCrashReporter::runCrashHandlers() {
  // No signal handling is done here on top of what AddSignalHandler() does:
  // on Windows the signal handling is implmented via
  // SetUnhandledExceptionFilter() which is thread-directed, and on Unix
  // platforms the handlers are only called for KillSigs out of which only
  // SIGQUIT seems to be process-directed and would be delivered to any thread
  // that is not blocking it, but if the thread it gets delivered to has a
  // ThreadCrashReporter installed during the interrupt — it seems reasonable to
  // let it run and print the thread's context information.

  // Call the reporters in LIFO order.
  ThreadCrashReporter *Reporter = CurrentReporter;
  while (Reporter) {
    Reporter->Callback();
    Reporter = Reporter->Next;
  }
}

ThreadCrashReporter::ThreadCrashReporter(SignalCallback ThreadLocalCallback)
    : Callback(std::move(ThreadLocalCallback)), Next(nullptr) {
  this->Next = CurrentReporter;
  CurrentReporter = this;
  // Don't reorder subsequent operations: whatever comes after might crash and
  // we want the the crash handler to see the reporter values we just set.
  std::atomic_signal_fence(std::memory_order_seq_cst);
}

ThreadCrashReporter::~ThreadCrashReporter() {
  assert(CurrentReporter == this);
  CurrentReporter = this->Next;
  // Don't reorder subsequent operations: whatever comes after might crash and
  // we want the the crash handler to see the reporter values we just set.
  std::atomic_signal_fence(std::memory_order_seq_cst);
}

} // namespace clangd
} // namespace clang

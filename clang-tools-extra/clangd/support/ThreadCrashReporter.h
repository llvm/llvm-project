//===--- ThreadCrashReporter.h - Thread local signal handling ----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_THREADCRASHREPORTER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_THREADCRASHREPORTER_H

#include "llvm/ADT/FunctionExtras.h"

namespace clang {
namespace clangd {

/// Allows setting per-thread abort/kill signal callbacks, to print additional
/// information about the crash depending on which thread got signalled.
class ThreadCrashReporter {
public:
  using SignalCallback = llvm::unique_function<void(void)>;

  /// Registers the callback as the first one in thread-local callback chain.
  ///
  /// Asserts if the current thread's callback is already set. The callback is
  /// likely to be invoked in a signal handler. Most LLVM signal handling is not
  /// strictly async-signal-safe. However reporters should avoid accessing data
  /// structures likely to be in a bad state on crash.
  ThreadCrashReporter(SignalCallback ThreadLocalCallback);
  /// Resets the current thread's callback to nullptr.
  ~ThreadCrashReporter();

  /// Moves are disabled to ease nesting and escaping considerations.
  ThreadCrashReporter(ThreadCrashReporter &&RHS) = delete;
  ThreadCrashReporter(const ThreadCrashReporter &) = delete;
  ThreadCrashReporter &operator=(ThreadCrashReporter &&) = delete;
  ThreadCrashReporter &operator=(const ThreadCrashReporter &) = delete;

  /// Calls all currently-active ThreadCrashReporters for the current thread.
  ///
  /// To be called from sys::AddSignalHandler() callback. Any signal filtering
  /// is the responsibility of the caller. While this function is intended to be
  /// called from signal handlers, it is not strictly async-signal-safe - see
  /// constructor comment.
  ///
  /// When several reporters are nested, the callbacks are called in LIFO order.
  static void runCrashHandlers();

private:
  SignalCallback Callback;
  /// Points to the next reporter up the stack.
  ThreadCrashReporter *Next;
};

} // namespace clangd
} // namespace clang

#endif

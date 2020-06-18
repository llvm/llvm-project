//===--- Shutdown.h - Unclean exit scenarios --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LSP specifies a protocol for shutting down: a `shutdown` request followed
// by an `exit` notification. If this protocol is followed, clangd should
// finish outstanding work and exit with code 0.
//
// The way this works in the happy case:
//  - when ClangdLSPServer gets `shutdown`, it sets a flag
//  - when ClangdLSPServer gets `exit`, it returns false to indicate end-of-LSP
//  - Transport::loop() returns with no error
//  - ClangdServer::run() checks the shutdown flag and returns with no error.
//  - we `return 0` from main()
//  - destructor of ClangdServer and other main()-locals runs.
//    This blocks until outstanding requests complete (results are ignored)
//  - global destructors run, such as fallback deletion of temporary files
//
// There are a number of things that can go wrong. Some are handled here, and
// some elsewhere.
//  - `exit` notification with no `shutdown`:
//    ClangdServer::run() sees this and returns false, main() returns nonzero.
//  - stdin/stdout are closed
//    The Transport detects this while doing IO and returns an error from loop()
//    ClangdServer::run() logs a message and then returns false, etc
//  - a request thread gets stuck, so the ClangdServer destructor hangs.
//    Before returning from main(), we start a watchdog thread to abort() the
//    process if it takes too long to exit. See abortAfterTimeout().
//  - clangd crashes (e.g. segfault or assertion)
//    A fatal signal is sent (SEGV, ABRT, etc)
//    The installed signal handler prints a stack trace and exits.
//  - parent process goes away or tells us to shut down
//    A "graceful shutdown" signal is sent (TERM, HUP, etc).
//    The installed signal handler calls requestShutdown() which sets a flag.
//    The Transport IO is interrupted, and Transport::loop() checks the flag and
//    returns an error, etc.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_SHUTDOWN_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_SHUTDOWN_H

#include <cerrno>
#include <chrono>

namespace clang {
namespace clangd {

/// Causes this process to crash if still running after Timeout.
void abortAfterTimeout(std::chrono::seconds Timeout);

/// Sets a flag to indicate that clangd was sent a shutdown signal, and the
/// transport loop should exit at the next opportunity.
/// If shutdown was already requested, aborts the process.
/// This function is threadsafe and signal-safe.
void requestShutdown();
/// Checks whether requestShutdown() was called.
/// This function is threadsafe and signal-safe.
bool shutdownRequested();

/// Retry an operation if it gets interrupted by a signal.
/// This is like llvm::sys::RetryAfterSignal, except that if shutdown was
/// requested (which interrupts IO), we'll fail rather than retry.
template <typename Fun, typename Ret = decltype(std::declval<Fun>()())>
Ret retryAfterSignalUnlessShutdown(
    const std::enable_if_t<true, Ret> &Fail, // Suppress deduction.
    const Fun &F) {
  Ret Res;
  do {
    if (shutdownRequested())
      return Fail;
    errno = 0;
    Res = F();
  } while (Res == Fail && errno == EINTR);
  return Res;
}

} // namespace clangd
} // namespace clang

#endif

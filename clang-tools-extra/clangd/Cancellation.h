//===--- Cancellation.h -------------------------------------------*-C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Cancellation mechanism for long-running tasks.
//
// This manages interactions between:
//
// 1. Client code that starts some long-running work, and maybe cancels later.
//
//   std::pair<Context, Canceler> Task = cancelableTask();
//   {
//     WithContext Cancelable(std::move(Task.first));
//     Expected
//     deepThoughtAsync([](int answer){ errs() << answer; });
//   }
//   // ...some time later...
//   if (User.fellAsleep())
//     Task.second();
//
//  (This example has an asynchronous computation, but synchronous examples
//  work similarly - the Canceler should be invoked from another thread).
//
// 2. Library code that executes long-running work, and can exit early if the
//   result is not needed.
//
//   void deepThoughtAsync(std::function<void(int)> Callback) {
//     runAsync([Callback]{
//       int A = ponder(6);
//       if (isCancelled())
//         return;
//       int B = ponder(9);
//       if (isCancelled())
//         return;
//       Callback(A * B);
//     });
//   }
//
//   (A real example may invoke the callback with an error on cancellation,
//   the CancelledError is provided for this purpose).
//
// Cancellation has some caveats:
//   - the work will only stop when/if the library code next checks for it.
//     Code outside clangd such as Sema will not do this.
//   - it's inherently racy: client code must be prepared to accept results
//     even after requesting cancellation.
//   - it's Context-based, so async work must be dispatched to threads in
//     ways that preserve the context. (Like runAsync() or TUScheduler).
//
// FIXME: We could add timestamps to isCancelled() and CancelledError.
//        Measuring the start -> cancel -> acknowledge -> finish timeline would
//        help find where libraries' cancellation should be improved.

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CANCELLATION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CANCELLATION_H

#include "Context.h"
#include "llvm/Support/Error.h"
#include <functional>
#include <system_error>

namespace clang {
namespace clangd {

/// A canceller requests cancellation of a task, when called.
/// Calling it again has no effect.
using Canceler = std::function<void()>;

/// Defines a new task whose cancellation may be requested.
/// The returned Context defines the scope of the task.
/// When the context is active, isCancelled() is false until the Canceler is
/// invoked, and true afterwards.
std::pair<Context, Canceler> cancelableTask();

/// True if the current context is within a cancelable task which was cancelled.
/// Always false if there is no active cancelable task.
/// This isn't free (context lookup) - don't call it in a tight loop.
bool isCancelled(const Context &Ctx = Context::current());

/// Conventional error when no result is returned due to cancellation.
class CancelledError : public llvm::ErrorInfo<CancelledError> {
public:
  static char ID;

  void log(llvm::raw_ostream &OS) const override {
    OS << "Task was cancelled.";
  }
  std::error_code convertToErrorCode() const override {
    return std::make_error_code(std::errc::operation_canceled);
  }
};

} // namespace clangd
} // namespace clang

#endif

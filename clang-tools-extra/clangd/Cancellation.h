//===--- Cancellation.h -------------------------------------------*-C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Cancellation mechanism for async tasks. Roughly all the clients of this code
// can be classified into three categories:
// 1. The code that creates and schedules async tasks, e.g. TUScheduler.
// 2. The callers of the async method that can cancel some of the running tasks,
// e.g. `ClangdLSPServer`
// 3. The code running inside the async task itself, i.e. code completion or
// find definition implementation that run clang, etc.
//
// For (1), the guideline is to accept a callback for the result of async
// operation and return a `TaskHandle` to allow cancelling the request.
//
//  TaskHandle someAsyncMethod(Runnable T,
//  function<void(llvm::Expected<ResultType>)> Callback) {
//   auto TH = Task::createHandle();
//   WithContext ContextWithCancellationToken(TH);
//   auto run = [](){
//     Callback(T());
//   }
//   // Start run() in a new async thread, and make sure to propagate Context.
//   return TH;
// }
//
// The callers of async methods (2) can issue cancellations and should be
// prepared to handle `TaskCancelledError` result:
//
// void Caller() {
//   // You should store this handle if you wanna cancel the task later on.
//   TaskHandle TH = someAsyncMethod(Task, [](llvm::Expected<ResultType> R) {
//     if(/*check for task cancellation error*/)
//       // Handle the error
//     // Do other things on R.
//   });
//   // To cancel the task:
//   sleep(5);
//   TH->cancel();
// }
//
// The worker code itself (3) should check for cancellations using
// `Task::isCancelled` that can be retrieved via `getCurrentTask()`.
//
// llvm::Expected<ResultType> AsyncTask() {
//    // You can either store the read only TaskHandle by calling getCurrentTask
//    // once and just use the variable everytime you want to check for
//    // cancellation, or call isCancelled everytime. The former is more
//    // efficient if you are going to have multiple checks.
//    const auto T = getCurrentTask();
//    // DO SMTHNG...
//    if(T.isCancelled()) {
//      // Task has been cancelled, lets get out.
//      return llvm::makeError<CancelledError>();
//    }
//    // DO SOME MORE THING...
//    if(T.isCancelled()) {
//      // Task has been cancelled, lets get out.
//      return llvm::makeError<CancelledError>();
//    }
//    return ResultType(...);
// }
// If the operation was cancelled before task could run to completion, it should
// propagate the TaskCancelledError as a result.

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CANCELLATION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CANCELLATION_H

#include "Context.h"
#include "llvm/Support/Error.h"
#include <atomic>
#include <memory>
#include <system_error>

namespace clang {
namespace clangd {

/// Enables signalling a cancellation on an async task or checking for
/// cancellation. It is thread-safe to trigger cancellation from multiple
/// threads or check for cancellation. Task object for the currently running
/// task can be obtained via clangd::getCurrentTask().
class Task {
public:
  void cancel() { CT = true; }
  /// If cancellation checks are rare, one could use the isCancelled() helper in
  /// the namespace to simplify the code. However, if cancellation checks are
  /// frequent, the guideline is first obtain the Task object for the currently
  /// running task with getCurrentTask() and do cancel checks using it to avoid
  /// extra lookups in the Context.
  bool isCancelled() const { return CT; }

  /// Creates a task handle that can be used by an async task to check for
  /// information that can change during it's runtime, like Cancellation.
  static std::shared_ptr<Task> createHandle() {
    return std::shared_ptr<Task>(new Task());
  }

  Task(const Task &) = delete;
  Task &operator=(const Task &) = delete;
  Task(Task &&) = delete;
  Task &operator=(Task &&) = delete;

private:
  Task() : CT(false) {}
  std::atomic<bool> CT;
};
using ConstTaskHandle = std::shared_ptr<const Task>;
using TaskHandle = std::shared_ptr<Task>;

/// Fetches current task information from Context. TaskHandle must have been
/// stashed into context beforehand.
const Task &getCurrentTask();

/// Stashes current task information within the context.
LLVM_NODISCARD Context setCurrentTask(ConstTaskHandle TH);

/// Checks whether the current task has been cancelled or not.
/// Consider storing the task handler returned by getCurrentTask and then
/// calling isCancelled through it. getCurrentTask is expensive since it does a
/// lookup in the context.
inline bool isCancelled() { return getCurrentTask().isCancelled(); }

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

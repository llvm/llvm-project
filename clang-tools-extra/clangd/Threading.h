//===--- ThreadPool.h --------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_THREADING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_THREADING_H

#include "Context.h"
#include "Function.h"
#include "llvm/ADT/Twine.h"
#include <cassert>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

namespace clang {
namespace clangd {

/// A threadsafe flag that is initially clear.
class Notification {
public:
  // Sets the flag. No-op if already set.
  void notify();
  // Blocks until flag is set.
  void wait() const;

private:
  bool Notified = false;
  mutable std::condition_variable CV;
  mutable std::mutex Mu;
};

/// Limits the number of threads that can acquire the lock at the same time.
class Semaphore {
public:
  Semaphore(std::size_t MaxLocks);

  void lock();
  void unlock();

private:
  std::mutex Mutex;
  std::condition_variable SlotsChanged;
  std::size_t FreeSlots;
};

/// A point in time we may wait for, or None to wait forever.
/// (Not time_point::max(), because many std::chrono implementations overflow).
using Deadline = llvm::Optional<std::chrono::steady_clock::time_point>;
/// Makes a deadline from a timeout in seconds.
Deadline timeoutSeconds(llvm::Optional<double> Seconds);
/// Waits on a condition variable until F() is true or D expires.
template <typename Func>
LLVM_NODISCARD bool wait(std::unique_lock<std::mutex> &Lock,
                         std::condition_variable &CV, Deadline D, Func F) {
  if (D)
    return CV.wait_until(Lock, *D, F);
  CV.wait(Lock, F);
  return true;
}

/// Runs tasks on separate (detached) threads and wait for all tasks to finish.
/// Objects that need to spawn threads can own an AsyncTaskRunner to ensure they
/// all complete on destruction.
class AsyncTaskRunner {
public:
  /// Destructor waits for all pending tasks to finish.
  ~AsyncTaskRunner();

  void wait() const { (void) wait(llvm::None); }
  LLVM_NODISCARD bool wait(Deadline D) const;
  // The name is used for tracing and debugging (e.g. to name a spawned thread).
  void runAsync(llvm::Twine Name, UniqueFunction<void()> Action);

private:
  mutable std::mutex Mutex;
  mutable std::condition_variable TasksReachedZero;
  std::size_t InFlightTasks = 0;
};
} // namespace clangd
} // namespace clang
#endif

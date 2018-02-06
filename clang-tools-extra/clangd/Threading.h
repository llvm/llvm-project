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
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

namespace clang {
namespace clangd {

/// A shared boolean flag indicating if the computation was cancelled.
/// Once cancelled, cannot be returned to the previous state.
class CancellationFlag {
public:
  CancellationFlag();

  void cancel() {
    assert(WasCancelled && "the object was moved");
    WasCancelled->store(true);
  }

  bool isCancelled() const {
    assert(WasCancelled && "the object was moved");
    return WasCancelled->load();
  }

private:
  std::shared_ptr<std::atomic<bool>> WasCancelled;
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

/// Runs tasks on separate (detached) threads and wait for all tasks to finish.
/// Objects that need to spawn threads can own an AsyncTaskRunner to ensure they
/// all complete on destruction.
class AsyncTaskRunner {
public:
  /// Destructor waits for all pending tasks to finish.
  ~AsyncTaskRunner();

  void waitForAll();
  void runAsync(UniqueFunction<void()> Action);

private:
  std::mutex Mutex;
  std::condition_variable TasksReachedZero;
  std::size_t InFlightTasks = 0;
};
} // namespace clangd
} // namespace clang
#endif

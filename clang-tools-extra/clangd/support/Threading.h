//===--- Threading.h - Abstractions for multithreading -----------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_THREADING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_THREADING_H

#include "support/Context.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/Twine.h"
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
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

  bool try_lock();
  void lock();
  void unlock();

private:
  std::mutex Mutex;
  std::condition_variable SlotsChanged;
  std::size_t FreeSlots;
};

/// A point in time we can wait for.
/// Can be zero (don't wait) or infinity (wait forever).
/// (Not time_point::max(), because many std::chrono implementations overflow).
class Deadline {
public:
  Deadline(std::chrono::steady_clock::time_point Time)
      : Type(Finite), Time(Time) {}
  static Deadline zero() { return Deadline(Zero); }
  static Deadline infinity() { return Deadline(Infinite); }

  std::chrono::steady_clock::time_point time() const {
    assert(Type == Finite);
    return Time;
  }
  bool expired() const {
    return (Type == Zero) ||
           (Type == Finite && Time < std::chrono::steady_clock::now());
  }
  bool operator==(const Deadline &Other) const {
    return (Type == Other.Type) && (Type != Finite || Time == Other.Time);
  }

private:
  enum Type { Zero, Infinite, Finite };

  Deadline(enum Type Type) : Type(Type) {}
  enum Type Type;
  std::chrono::steady_clock::time_point Time;
};

/// Makes a deadline from a timeout in seconds. None means wait forever.
Deadline timeoutSeconds(llvm::Optional<double> Seconds);
/// Wait once on CV for the specified duration.
void wait(std::unique_lock<std::mutex> &Lock, std::condition_variable &CV,
          Deadline D);
/// Waits on a condition variable until F() is true or D expires.
template <typename Func>
LLVM_NODISCARD bool wait(std::unique_lock<std::mutex> &Lock,
                         std::condition_variable &CV, Deadline D, Func F) {
  while (!F()) {
    if (D.expired())
      return false;
    wait(Lock, CV, D);
  }
  return true;
}

/// Runs tasks on separate (detached) threads and wait for all tasks to finish.
/// Objects that need to spawn threads can own an AsyncTaskRunner to ensure they
/// all complete on destruction.
class AsyncTaskRunner {
public:
  /// Destructor waits for all pending tasks to finish.
  ~AsyncTaskRunner();

  void wait() const { (void)wait(Deadline::infinity()); }
  LLVM_NODISCARD bool wait(Deadline D) const;
  // The name is used for tracing and debugging (e.g. to name a spawned thread).
  void runAsync(const llvm::Twine &Name, llvm::unique_function<void()> Action);

private:
  mutable std::mutex Mutex;
  mutable std::condition_variable TasksReachedZero;
  std::size_t InFlightTasks = 0;
};

/// Runs \p Action asynchronously with a new std::thread. The context will be
/// propagated.
template <typename T>
std::future<T> runAsync(llvm::unique_function<T()> Action) {
  return std::async(
      std::launch::async,
      [](llvm::unique_function<T()> &&Action, Context Ctx) {
        WithContext WithCtx(std::move(Ctx));
        return Action();
      },
      std::move(Action), Context::current().clone());
}

/// Memoize is a cache to store and reuse computation results based on a key.
///
///   Memoize<DenseMap<int, bool>> PrimeCache;
///   for (int I : RepetitiveNumbers)
///     if (PrimeCache.get(I, [&] { return expensiveIsPrime(I); }))
///       llvm::errs() << "Prime: " << I << "\n";
///
/// The computation will only be run once for each key.
/// This class is threadsafe. Concurrent calls for the same key may run the
/// computation multiple times, but each call will return the same result.
template <typename Container> class Memoize {
  mutable Container Cache;
  std::unique_ptr<std::mutex> Mu;

public:
  Memoize() : Mu(std::make_unique<std::mutex>()) {}

  template <typename T, typename Func>
  typename Container::mapped_type get(T &&Key, Func Compute) const {
    {
      std::lock_guard<std::mutex> Lock(*Mu);
      auto It = Cache.find(Key);
      if (It != Cache.end())
        return It->second;
    }
    // Don't hold the mutex while computing.
    auto V = Compute();
    {
      std::lock_guard<std::mutex> Lock(*Mu);
      auto R = Cache.try_emplace(std::forward<T>(Key), V);
      // Insert into cache may fail if we raced with another thread.
      if (!R.second)
        return R.first->second; // Canonical value, from other thread.
    }
    return V;
  }
};

/// Used to guard an operation that should run at most every N seconds.
///
/// Usage:
///   mutable PeriodicThrottler ShouldLog(std::chrono::seconds(1));
///   void calledFrequently() {
///     if (ShouldLog())
///       log("this is not spammy");
///   }
///
/// This class is threadsafe. If multiple threads are involved, then the guarded
/// operation still needs to be threadsafe!
class PeriodicThrottler {
  using Stopwatch = std::chrono::steady_clock;
  using Rep = Stopwatch::duration::rep;

  Rep Period;
  std::atomic<Rep> Next;

public:
  /// If Period is zero, the throttler will return true every time.
  PeriodicThrottler(Stopwatch::duration Period, Stopwatch::duration Delay = {})
      : Period(Period.count()),
        Next((Stopwatch::now() + Delay).time_since_epoch().count()) {}

  /// Returns whether the operation should run at this time.
  /// operator() is safe to call concurrently.
  bool operator()();
};

} // namespace clangd
} // namespace clang
#endif

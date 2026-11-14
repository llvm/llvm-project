//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17
// UNSUPPORTED: no-threads
// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// void __apply(size_t chunk_count, void* context, void (*func)(void* context, size_t chunk)) noexcept;

// Checks that a function applied via __apply may hold a mutex while making a nested call to __apply.

#include <__pstl/backends/statpart.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <thread>

// A mutex that detects an attempt to lock it by the thread that already holds it, instead of deadlocking.
class SelfDeadlockDetectingMutex {
public:
  void lock() {
    assert(owner_.load() != std::this_thread::get_id() && "deadlock: the mutex is already held by this thread");
    mutex_.lock();
    owner_.store(std::this_thread::get_id());
  }

  void unlock() {
    owner_.store(std::thread::id{});
    mutex_.unlock();
  }

private:
  std::mutex mutex_;
  std::atomic<std::thread::id> owner_;
};

int main(int, char**) {
  SelfDeadlockDetectingMutex mutex;
  for (int round = 0; round < 10; ++round) {
    // Run the outer task in parallel.
    std::__pstl::__statpart::__apply(8, &mutex, [](void* context, std::size_t) {
      // Lock the mutex to detect potential deadlocks if the following invocation of __apply
      // would try participating in the outer task from within the same thread.
      std::lock_guard<SelfDeadlockDetectingMutex> lock(*static_cast<SelfDeadlockDetectingMutex*>(context));
      // Run the inner task in parallel.
      std::__pstl::__statpart::__apply(32, nullptr, [](void*, std::size_t) {
        // Keep the nested tasks in flight long enough for the waiting thread to look for other work.
        std::this_thread::sleep_for(std::chrono::microseconds(20));
      });
    });
  }
  return 0;
}

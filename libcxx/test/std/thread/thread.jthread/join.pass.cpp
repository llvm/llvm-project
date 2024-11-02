//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: availability-synchronization_library-missing

// void join();

#include <atomic>
#include <cassert>
#include <chrono>
#include <concepts>
#include <functional>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

int main(int, char**) {
  // Effects: Blocks until the thread represented by *this has completed.
  {
    std::atomic_int calledTimes = 0;
    std::vector<std::jthread> jts;
    constexpr auto numberOfThreads = 10u;
    jts.reserve(numberOfThreads);
    for (auto i = 0u; i < numberOfThreads; ++i) {
      jts.emplace_back(support::make_test_jthread([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        calledTimes.fetch_add(1, std::memory_order_relaxed);
      }));
    }

    for (auto i = 0u; i < numberOfThreads; ++i) {
      jts[i].join();
    }

    // If join did block, calledTimes must equal to numberOfThreads
    // If join did not block, there is a chance that the check below happened
    // before test threads incrementing the counter, thus calledTimed would
    // be less than numberOfThreads.
    // This is not going to catch issues 100%. Creating more threads to increase
    // the probability of catching the issue
    assert(calledTimes.load(std::memory_order_relaxed) == numberOfThreads);
  }

  // Synchronization: The completion of the thread represented by *this synchronizes with
  // ([intro.multithread]) the corresponding successful join() return.
  {
    bool flag       = false;
    std::jthread jt = support::make_test_jthread([&] { flag = true; });
    jt.join();
    assert(flag); // non atomic write is visible to the current thread
  }

  // Postconditions: The thread represented by *this has completed. get_id() == id().
  {
    std::jthread jt = support::make_test_jthread([] {});
    assert(jt.get_id() != std::jthread::id());
    jt.join();
    assert(jt.get_id() == std::jthread::id());
  }

#if !defined(TEST_HAS_NO_EXCEPTIONS)
  // Throws: system_error when an exception is required ([thread.req.exception]).
  // invalid_argument - if the thread is not joinable.
  {
    std::jthread jt;
    try {
      jt.join();
      assert(false);
    } catch (const std::system_error& err) {
      assert(err.code() == std::errc::invalid_argument);
    }
  }

#endif

  return 0;
}

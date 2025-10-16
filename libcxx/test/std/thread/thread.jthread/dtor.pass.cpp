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

// ~jthread();

#include <atomic>
#include <cassert>
#include <optional>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

int main(int, char**) {
  // !joinable()
  {
    std::jthread jt;
    assert(!jt.joinable());
  }

  // If joinable() is true, calls request_stop() and then join().
  // request_stop is called
  {
    std::optional<std::jthread> jt = support::make_test_jthread([] {});
    bool called                    = false;
    std::stop_callback cb(jt->get_stop_token(), [&called] { called = true; });
    jt.reset();
    assert(called);
  }

  // If joinable() is true, calls request_stop() and then join().
  // join is called
  {
    std::atomic_int calledTimes = 0;
    std::vector<std::jthread> jts;

    constexpr auto numberOfThreads = 10u;
    jts.reserve(numberOfThreads);
    for (auto i = 0u; i < numberOfThreads; ++i) {
      jts.emplace_back(support::make_test_jthread([&calledTimes] {
        std::this_thread::sleep_for(std::chrono::milliseconds{2});
        calledTimes.fetch_add(1, std::memory_order_relaxed);
      }));
    }
    jts.clear();

    // If join was called as expected, calledTimes must equal to numberOfThreads
    // If join was not called, there is a chance that the check below happened
    // before test threads incrementing the counter, thus calledTimed would
    // be less than numberOfThreads.
    // This is not going to catch issues 100%. Creating more threads would increase
    // the probability of catching the issue
    assert(calledTimes.load(std::memory_order_relaxed) == numberOfThreads);
  }

  return 0;
}

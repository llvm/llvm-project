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
// ADDITIONAL_COMPILE_FLAGS(gcc-style-warnings): -Wno-self-move

// jthread& operator=(jthread&&) noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

static_assert(std::is_nothrow_move_assignable_v<std::jthread>);

int main(int, char**) {
  // If &x == this is true, there are no effects.
  {
    std::jthread j = support::make_test_jthread([] {});
    auto id        = j.get_id();
    auto ssource   = j.get_stop_source();
    j              = std::move(j);
    assert(j.get_id() == id);
    assert(j.get_stop_source() == ssource);
  }

  // if joinable() is true, calls request_stop() and then join()
  // request_stop is called
  {
    std::jthread j1 = support::make_test_jthread([] {});
    bool called     = false;
    std::stop_callback cb(j1.get_stop_token(), [&called] { called = true; });

    std::jthread j2 = support::make_test_jthread([] {});
    j1              = std::move(j2);
    assert(called);
  }

  // if joinable() is true, calls request_stop() and then join()
  // join is called
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
      jts[i] = std::jthread{};
    }

    // If join was called as expected, calledTimes must equal to numberOfThreads
    // If join was not called, there is a chance that the check below happened
    // before test threads incrementing the counter, thus calledTimed would
    // be less than numberOfThreads.
    // This is not going to catch issues 100%. Creating more threads to increase
    // the probability of catching the issue
    assert(calledTimes.load(std::memory_order_relaxed) == numberOfThreads);
  }

  // then assigns the state of x to *this
  {
    std::jthread j1 = support::make_test_jthread([] {});
    std::jthread j2 = support::make_test_jthread([] {});
    auto id2        = j2.get_id();
    auto ssource2   = j2.get_stop_source();

    j1 = std::move(j2);

    assert(j1.get_id() == id2);
    assert(j1.get_stop_source() == ssource2);
  }

  // sets x to a default constructed state
  {
    std::jthread j1 = support::make_test_jthread([] {});
    std::jthread j2 = support::make_test_jthread([] {});
    j1              = std::move(j2);

    assert(j2.get_id() == std::jthread::id());
    assert(!j2.get_stop_source().stop_possible());
  }

  // joinable is false
  {
    std::jthread j1;
    std::jthread j2 = support::make_test_jthread([] {});

    auto j2Id = j2.get_id();

    j1 = std::move(j2);

    assert(j1.get_id() == j2Id);
  }

  // LWG3788: self-assignment
  {
    std::jthread j = support::make_test_jthread([] {});
    auto oldId     = j.get_id();
    j              = std::move(j);

    assert(j.get_id() == oldId);
  }

  return 0;
}

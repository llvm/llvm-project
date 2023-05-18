//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: libcpp-has-no-experimental-stop_token
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: availability-synchronization_library-missing

// [[nodiscard]] bool stop_requested() const noexcept;
// true if *this has ownership of a stop state that has received a stop request; otherwise, false.

#include <cassert>
#include <chrono>
#include <concepts>
#include <optional>
#include <stop_token>
#include <type_traits>

#include "make_test_thread.h"
#include "test_macros.h"

template <class T>
concept IsStopRequestedNoexcept = requires(const T& t) {
  { t.stop_requested() } noexcept;
};

static_assert(IsStopRequestedNoexcept<std::stop_source>);

int main(int, char**) {
  // no state
  {
    const std::stop_source ss{std::nostopstate};
    assert(!ss.stop_requested());
  }

  // has state
  {
    std::stop_source ss;
    assert(!ss.stop_requested());

    ss.request_stop();
    assert(ss.stop_requested());
  }

  // request from another instance with same state
  {
    std::stop_source ss1;
    auto ss2 = ss1;
    ss2.request_stop();
    assert(ss1.stop_requested());
  }

  // request from another instance with different state
  {
    std::stop_source ss1;
    std::stop_source ss2;

    ss2.request_stop();
    assert(!ss1.stop_requested());
  }

  // multiple threads
  {
    std::stop_source ss;

    std::thread t = support::make_test_thread([&]() { ss.request_stop(); });

    t.join();
    assert(ss.stop_requested());
  }

  // [thread.stopsource.intro] A call to request_stop that returns true
  // synchronizes with a call to stop_requested on an associated stop_source
  // or stop_source object that returns true.
  {
    std::stop_source ss;

    bool flag = false;

    std::thread t = support::make_test_thread([&]() {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(1ms);

      // happens-before request_stop
      flag   = true;
      auto b = ss.request_stop();
      assert(b);
    });

    while (!ss.stop_requested()) {
      std::this_thread::yield();
    }

    // write should be visible to the current thread
    assert(flag == true);

    t.join();
  }

  return 0;
}

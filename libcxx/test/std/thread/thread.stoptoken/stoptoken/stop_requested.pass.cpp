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
// Returns: true if *this has ownership of a stop state that has received a stop request; otherwise, false.

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

static_assert(IsStopRequestedNoexcept<std::stop_token>);

int main(int, char**) {
  // no state
  {
    const std::stop_token st;
    assert(!st.stop_requested());
  }

  // has state
  {
    std::stop_source ss;
    const auto st = ss.get_token();
    assert(!st.stop_requested());

    ss.request_stop();
    assert(st.stop_requested());
  }

  // already requested before constructor
  {
    std::stop_source ss;
    ss.request_stop();
    const auto st = ss.get_token();
    assert(st.stop_requested());
  }

  // stop_token should share the state
  {
    std::optional<std::stop_source> ss{std::in_place};
    ss->request_stop();
    const auto st = ss->get_token();

    ss.reset();
    assert(st.stop_requested());
  }

  // single stop_source, multiple stop_token
  {
    std::stop_source ss;
    const auto st1 = ss.get_token();
    const auto st2 = ss.get_token();
    assert(!st1.stop_requested());
    assert(!st2.stop_requested());

    ss.request_stop();
    assert(st1.stop_requested());
    assert(st2.stop_requested());
  }

  // multiple stop_source, multiple stop_token
  {
    std::stop_source ss1;
    std::stop_source ss2;

    const auto st1 = ss1.get_token();
    const auto st2 = ss2.get_token();
    assert(!st1.stop_requested());
    assert(!st2.stop_requested());

    ss1.request_stop();
    assert(st1.stop_requested());
    assert(!st2.stop_requested());
  }

  // multiple threads
  {
    std::stop_source ss;
    const auto st = ss.get_token();
    assert(!st.stop_requested());

    std::thread t = support::make_test_thread([&]() { ss.request_stop(); });

    t.join();
    assert(st.stop_requested());
  }

  // maybe concurrent calls
  {
    std::stop_source ss;
    const auto st = ss.get_token();
    assert(!st.stop_requested());

    std::thread t = support::make_test_thread([&]() { ss.request_stop(); });

    while (!st.stop_requested()) {
      // should eventually exit the loop
      std::this_thread::yield();
    }

    t.join();
  }

  // [thread.stoptoken.intro] A call to request_stop that returns true
  // synchronizes with a call to stop_requested on an associated stop_token
  // or stop_source object that returns true.
  {
    std::stop_source ss;
    const auto st = ss.get_token();
    assert(!st.stop_requested());

    bool flag = false;

    std::thread t = support::make_test_thread([&]() {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(1ms);

      // happens-before request_stop
      flag   = true;
      auto b = ss.request_stop();
      assert(b);
    });

    while (!st.stop_requested()) {
      std::this_thread::yield();
    }

    // write should be visible to the current thread
    assert(flag == true);

    t.join();
  }

  return 0;
}

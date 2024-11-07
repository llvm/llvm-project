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

// [[nodiscard]] bool stop_possible() const noexcept;
// Returns: false if:
//    - *this does not have ownership of a stop state, or
//    - a stop request was not made and there are no associated stop_source objects;
// otherwise, true.

#include <cassert>
#include <concepts>
#include <optional>
#include <stop_token>
#include <type_traits>

#include "make_test_thread.h"
#include "test_macros.h"

template <class T>
concept IsStopPossibleNoexcept = requires(const T& t) {
  { t.stop_possible() } noexcept;
};

static_assert(IsStopPossibleNoexcept<std::stop_token>);

int main(int, char**) {
  // no state
  {
    const std::stop_token st;
    assert(!st.stop_possible());
  }

  // a stop request was not made and there are no associated stop_source objects
  {
    std::optional<std::stop_source> ss{std::in_place};
    const auto st = ss->get_token();
    ss.reset();

    assert(!st.stop_possible());
  }

  // a stop request was not made, but there is an associated stop_source objects
  {
    std::stop_source ss;
    const auto st = ss.get_token();
    assert(st.stop_possible());
  }

  // a stop request was made and there are no associated stop_source objects
  {
    std::optional<std::stop_source> ss{std::in_place};
    const auto st = ss->get_token();
    ss->request_stop();
    ss.reset();

    assert(st.stop_possible());
  }

  // a stop request was made and there is an associated stop_source objects
  {
    std::stop_source ss;
    const auto st = ss.get_token();
    ss.request_stop();
    assert(st.stop_possible());
  }

  // a stop request was made on a different thread and
  // there are no associated stop_source objects
  {
    std::optional<std::stop_source> ss{std::in_place};
    const auto st = ss->get_token();

    std::thread t = support::make_test_thread([&]() {
      ss->request_stop();
      ss.reset();
    });

    assert(st.stop_possible());
    t.join();
    assert(st.stop_possible());

  }

  return 0;
}

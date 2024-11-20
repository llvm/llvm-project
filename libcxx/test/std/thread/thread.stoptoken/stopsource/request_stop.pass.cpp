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

// bool request_stop() noexcept;

#include <cassert>
#include <chrono>
#include <concepts>
#include <optional>
#include <stop_token>
#include <type_traits>

#include "make_test_thread.h"
#include "test_macros.h"

template <class T>
concept IsRequestStopNoexcept = requires(T& t) {
  { t.request_stop() } noexcept;
};

static_assert(IsRequestStopNoexcept<std::stop_source>);

int main(int, char**) {
  // If *this does not have ownership of a stop state, returns false
  {
    std::stop_source ss{std::nostopstate};
    auto ret = ss.request_stop();
    assert(!ret);
    assert(!ss.stop_requested());
  }

  // Otherwise, atomically determines whether the owned stop state has received
  // a stop request, and if not, makes a stop request
  {
    std::stop_source ss;

    auto ret = ss.request_stop();
    assert(ret);
    assert(ss.stop_requested());
  }

  // already requested
  {
    std::stop_source ss;
    ss.request_stop();
    assert(ss.stop_requested());

    auto ret = ss.request_stop();
    assert(!ret);
    assert(ss.stop_requested());
  }

  // If the request was made, the callbacks registered by
  // associated stop_callback objects are synchronously called.
  {
    std::stop_source ss;
    auto st = ss.get_token();

    bool cb1Called = false;
    bool cb2Called = false;
    std::stop_callback sc1(st, [&] { cb1Called = true; });
    std::stop_callback sc2(st, [&] { cb2Called = true; });

    ss.request_stop();
    assert(cb1Called);
    assert(cb2Called);
  }

  return 0;
}

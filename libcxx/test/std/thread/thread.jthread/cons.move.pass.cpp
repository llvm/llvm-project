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

// jthread(jthread&& x) noexcept;

#include <cassert>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>

#include "make_test_thread.h"
#include "test_macros.h"

static_assert(std::is_nothrow_move_constructible_v<std::jthread>);

int main(int, char**) {
  {
    // x.get_id() == id() and get_id() returns the value of x.get_id() prior
    // to the start of construction.
    std::jthread j1 = support::make_test_jthread([] {});
    auto id1        = j1.get_id();

    std::jthread j2(std::move(j1));
    assert(j1.get_id() == std::jthread::id());
    assert(j2.get_id() == id1);
  }

  {
    // ssource has the value of x.ssource prior to the start of construction
    // and x.ssource.stop_possible() is false.
    std::jthread j1 = support::make_test_jthread([] {});
    auto ss1        = j1.get_stop_source();

    std::jthread j2(std::move(j1));
    assert(ss1 == j2.get_stop_source());
    assert(!j1.get_stop_source().stop_possible());
    assert(j2.get_stop_source().stop_possible());
  }

  return 0;
}

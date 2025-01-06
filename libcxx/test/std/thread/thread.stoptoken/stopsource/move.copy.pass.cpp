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

// stop_source& operator=(stop_source&& rhs) noexcept;

#include <cassert>
#include <concepts>
#include <stop_token>
#include <type_traits>
#include <utility>

#include "test_macros.h"

static_assert(std::is_nothrow_move_assignable_v<std::stop_source>);

int main(int, char**) {
  // have two different states
  {
    std::stop_source ss1;
    std::stop_source ss2;

    assert(ss1 != ss2);

    ss2.request_stop();

    assert(!ss1.stop_requested());
    assert(ss2.stop_requested());

    std::same_as<std::stop_source&> decltype(auto) ref = ss1 = std::move(ss2);
    assert(&ref == &ss1);

    assert(ss1.stop_requested());
    assert(!ss2.stop_possible());
    assert(!ss2.stop_requested());
  }

  // this has no state
  {
    std::stop_source ss1{std::nostopstate};
    std::stop_source ss2;

    assert(ss1 != ss2);

    ss2.request_stop();

    assert(!ss1.stop_requested());
    assert(!ss1.stop_possible());
    assert(ss2.stop_requested());
    assert(ss2.stop_possible());

    std::same_as<std::stop_source&> decltype(auto) ref = ss1 = std::move(ss2);
    assert(&ref == &ss1);

    assert(ss1.stop_requested());
    assert(ss1.stop_possible());
    assert(!ss2.stop_requested());
    assert(!ss2.stop_possible());
  }

  // other has no state
  {
    std::stop_source ss1;
    std::stop_source ss2{std::nostopstate};

    assert(ss1 != ss2);

    ss1.request_stop();

    assert(ss1.stop_requested());
    assert(ss1.stop_possible());
    assert(!ss2.stop_requested());
    assert(!ss2.stop_possible());

    std::same_as<std::stop_source&> decltype(auto) ref = ss1 = std::move(ss2);
    assert(&ref == &ss1);

    assert(ss1 == ss2);
    assert(!ss1.stop_requested());
    assert(!ss1.stop_possible());
    assert(!ss2.stop_requested());
    assert(!ss2.stop_possible());
  }

  // both no state
  {
    std::stop_source ss1{std::nostopstate};
    std::stop_source ss2{std::nostopstate};

    assert(ss1 == ss2);

    assert(!ss1.stop_requested());
    assert(!ss1.stop_possible());
    assert(!ss2.stop_requested());
    assert(!ss2.stop_possible());

    std::same_as<std::stop_source&> decltype(auto) ref = ss1 = std::move(ss2);
    assert(&ref == &ss1);

    assert(ss1 == ss2);
    assert(!ss1.stop_requested());
    assert(!ss1.stop_possible());
    assert(!ss2.stop_requested());
    assert(!ss2.stop_possible());
  }

  // self assignment
  {
    std::stop_source ss;
    auto& self = ss;

    assert(!ss.stop_requested());

    std::same_as<std::stop_source&> decltype(auto) ref = ss = std::move(self);
    assert(&ref == &ss);

    assert(!ss.stop_requested());

    ss.request_stop();
    assert(ss.stop_requested());
  }

  return 0;
}

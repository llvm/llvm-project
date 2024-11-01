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

// friend void swap(jthread& x, jthread& y) noexcept;

#include <cassert>
#include <thread>
#include <type_traits>

#include "test_macros.h"

template <class T>
concept IsFreeSwapNoexcept = requires(T& a, T& b) {
  { swap(a, b) } noexcept;
};

static_assert(IsFreeSwapNoexcept<std::jthread>);

int main(int, char**) {
  // x is default constructed
  {
    std::jthread t1;
    std::jthread t2{[] {}};
    const auto originalId2 = t2.get_id();
    swap(t1, t2);

    assert(t1.get_id() == originalId2);
    assert(t2.get_id() == std::jthread::id());
  }

  // y is default constructed
  {
    std::jthread t1([] {});
    std::jthread t2{};
    const auto originalId1 = t1.get_id();
    swap(t1, t2);

    assert(t1.get_id() == std::jthread::id());
    assert(t2.get_id() == originalId1);
  }

  // both not default constructed
  {
    std::jthread t1([] {});
    std::jthread t2{[] {}};
    const auto originalId1 = t1.get_id();
    const auto originalId2 = t2.get_id();
    swap(t1, t2);

    assert(t1.get_id() == originalId2);
    assert(t2.get_id() == originalId1);
  }

  // both default constructed
  {
    std::jthread t1;
    std::jthread t2;
    swap(t1, t2);

    assert(t1.get_id() == std::jthread::id());
    assert(t2.get_id() == std::jthread::id());
  }

  return 0;
}

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

// void swap(jthread& x) noexcept;

#include <cassert>
#include <thread>
#include <type_traits>

#include "make_test_thread.h"
#include "test_macros.h"

template <class T>
concept IsMemberSwapNoexcept = requires(T& a, T& b) {
  { a.swap(b) } noexcept;
};

static_assert(IsMemberSwapNoexcept<std::jthread>);

int main(int, char**) {
  // this is default constructed
  {
    std::jthread t1;
    std::jthread t2        = support::make_test_jthread([] {});
    const auto originalId2 = t2.get_id();
    t1.swap(t2);

    assert(t1.get_id() == originalId2);
    assert(t2.get_id() == std::jthread::id());
  }

  // that is default constructed
  {
    std::jthread t1 = support::make_test_jthread([] {});
    std::jthread t2{};
    const auto originalId1 = t1.get_id();
    t1.swap(t2);

    assert(t1.get_id() == std::jthread::id());
    assert(t2.get_id() == originalId1);
  }

  // both not default constructed
  {
    std::jthread t1        = support::make_test_jthread([] {});
    std::jthread t2        = support::make_test_jthread([] {});
    const auto originalId1 = t1.get_id();
    const auto originalId2 = t2.get_id();
    t1.swap(t2);

    assert(t1.get_id() == originalId2);
    assert(t2.get_id() == originalId1);
  }

  // both default constructed
  {
    std::jthread t1;
    std::jthread t2;
    t1.swap(t2);

    assert(t1.get_id() == std::jthread::id());
    assert(t2.get_id() == std::jthread::id());
  }

  return 0;
}

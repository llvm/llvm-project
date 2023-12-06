//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// sentinel() = default;
// constexpr explicit sentinel(sentinel_t<Base> end);
// constexpr sentinel(sentinel<!Const> s)
//   requires Const && convertible_to<sentinel_t<V>, sentinel_t<Base>>;

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "../types.h"

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    // Test the default ctor.
    using TakeView = std::ranges::take_view<MoveOnlyView>;
    using Sentinel = std::ranges::sentinel_t<TakeView>;
    Sentinel s;
    TakeView tv = TakeView(MoveOnlyView(buffer), 4);
    assert(tv.begin() + 4 == s);
  }

  {
    // Test the conversion from "sentinel" to "sentinel-to-const".
    using TakeView = std::ranges::take_view<MoveOnlyView>;
    using Sentinel = std::ranges::sentinel_t<TakeView>;
    using ConstSentinel = std::ranges::sentinel_t<const TakeView>;
    static_assert(std::is_convertible_v<Sentinel, ConstSentinel>);
    TakeView tv = TakeView(MoveOnlyView(buffer), 4);
    Sentinel s = tv.end();
    ConstSentinel cs = s;
    cs = s;  // test assignment also
    assert(tv.begin() + 4 == s);
    assert(tv.begin() + 4 == cs);
    assert(std::as_const(tv).begin() + 4 == s);
    assert(std::as_const(tv).begin() + 4 == cs);
  }

  {
    // Test the constructor from "base-sentinel" to "sentinel".
    using TakeView = std::ranges::take_view<MoveOnlyView>;
    using Sentinel = std::ranges::sentinel_t<TakeView>;
    sentinel_wrapper<int*> sw1 = MoveOnlyView(buffer).end();
    static_assert( std::is_constructible_v<Sentinel, sentinel_wrapper<int*>>);
    static_assert(!std::is_convertible_v<sentinel_wrapper<int*>, Sentinel>);
    auto s = Sentinel(sw1);
    std::same_as<sentinel_wrapper<int*>> auto sw2 = s.base();
    assert(base(sw2) == base(sw1));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// friend constexpr bool operator==(const inner-iterator& x, const inner-iterator& y);
//   requires forward_range<Base>;
//
// friend constexpr bool operator==(const inner-iterator& x, default_sentinel_t);

#include <ranges>

#include <concepts>
#include <string_view>
#include "../types.h"

template <class Iter>
concept CanCallEquals = requires(const Iter& i) {
  i == i;
  i != i;
};

constexpr bool test() {
  // When `View` is a forward range, `inner-iterator` supports both overloads of `operator==`.
  {
    SplitViewForward v("abc def", " ");
    auto val = *v.begin();
    auto b = val.begin();
    std::same_as<std::default_sentinel_t> decltype(auto) e = val.end();

    // inner-iterator == inner-iterator
    {
      assert(b == b);
      assert(!(b != b));
    }

    // inner-iterator == default_sentinel
    {
      assert(!(b == e));
      assert(b != e);

      assert(!(b == std::default_sentinel));
      assert(b != std::default_sentinel);
    }
  }

  // When `View` is an input range, `inner-iterator only supports comparing an `inner-iterator` to the default sentinel.
  {
    SplitViewInput v("abc def", ' ');
    auto val = *v.begin();
    auto b = val.begin();
    std::same_as<std::default_sentinel_t> decltype(auto) e = val.end();

    static_assert(!CanCallEquals<decltype(b)>);

    assert(!(b == std::default_sentinel));
    assert(b != std::default_sentinel);
    assert(!(b == e));
    assert(b != e);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

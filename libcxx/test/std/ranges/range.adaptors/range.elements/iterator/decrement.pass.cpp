//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr iterator& operator--() requires bidirectional_range<Base>;
// constexpr iterator operator--(int) requires bidirectional_range<Base>;

#include <array>
#include <cassert>
#include <ranges>
#include <tuple>

#include "test_iterators.h"

template <class Iter>
concept CanPreDecrement = requires(Iter it) { --it; };

template <class Iter>
concept CanPostDecrement = requires(Iter it) { it--; };

template <class Iter, class Sent = sentinel_wrapper<Iter>>
constexpr void testOne() {
  using Range          = std::ranges::subrange<Iter, Sent>;
  std::tuple<int> ts[] = {{1}, {2}, {3}};
  auto ev              = Range{Iter{&ts[0]}, Sent{Iter{&ts[0] + 3}}} | std::views::elements<0>;

  using ElementIter = std::ranges::iterator_t<decltype(ev)>;

  if constexpr (!std::bidirectional_iterator<Iter>) {
    auto it = ev.begin();
    static_assert(!CanPreDecrement<decltype(it)>);
    static_assert(!CanPostDecrement<decltype(it)>);
  } else {
    // --i
    {
      auto it = ev.begin();
      static_assert(CanPreDecrement<decltype(it)>);

      ++it;
      assert(base(it.base()) == &ts[1]);

      decltype(auto) result = --it;

      static_assert(std::is_same_v<decltype(result), ElementIter&>);
      assert(&result == &it);

      assert(base(it.base()) == &ts[0]);
    }

    // i--
    {
      auto it = ev.begin();
      static_assert(CanPostDecrement<decltype(it)>);

      ++it;
      assert(base(it.base()) == &ts[1]);

      decltype(auto) result = it--;

      static_assert(std::is_same_v<decltype(result), ElementIter>);

      assert(base(it.base()) == &ts[0]);
      assert(base(result.base()) == &ts[1]);
    }
  }
}

constexpr bool test() {
  using Ptr = std::tuple<int>*;
  testOne<cpp20_input_iterator<Ptr>>();
  testOne<forward_iterator<Ptr>>();
  testOne<bidirectional_iterator<Ptr>>();
  testOne<random_access_iterator<Ptr>>();
  testOne<contiguous_iterator<Ptr>>();
  testOne<Ptr>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr iterator& operator++();
// constexpr void operator++(int);
// constexpr iterator operator++(int) requires forward_range<Base>;

#include <array>
#include <cassert>
#include <ranges>
#include <tuple>

#include "test_iterators.h"

template <class Iter, class Sent = sentinel_wrapper<Iter>>
constexpr void testOne() {
  using Range          = std::ranges::subrange<Iter, Sent>;
  std::tuple<int> ts[] = {{1}, {2}, {3}};
  auto ev              = Range{Iter{&ts[0]}, Sent{Iter{&ts[0] + 3}}} | std::views::elements<0>;

  using ElementIter = std::ranges::iterator_t<decltype(ev)>;

  // ++i
  {
    auto it               = ev.begin();
    decltype(auto) result = ++it;

    static_assert(std::is_same_v<decltype(result), ElementIter&>);
    assert(&result == &it);

    assert(base(it.base()) == &ts[1]);
  }

  // i++
  {
    if constexpr (std::forward_iterator<Iter>) {
      auto it               = ev.begin();
      decltype(auto) result = it++;

      static_assert(std::is_same_v<decltype(result), ElementIter>);

      assert(base(it.base()) == &ts[1]);
      assert(base(result.base()) == &ts[0]);
    } else {
      auto it = ev.begin();
      it++;

      static_assert(std::is_same_v<decltype(it++), void>);
      assert(base(it.base()) == &ts[1]);
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

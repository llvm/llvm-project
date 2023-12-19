//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// friend constexpr bool operator==(iterator const&, sentinel const&);

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <utility>
#include "test_iterators.h"
#include "../types.h"

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr void test() {
  using View = minimal_view<Iterator, Sentinel>;

  std::array<int, 5> array{0, 1, 2, 3, 4};

  {
    View v(Iterator(array.data()), Sentinel(Iterator(array.data() + array.size())));
    std::ranges::filter_view view(std::move(v), AlwaysTrue{});
    auto const it = view.begin();
    auto const sent = view.end();
    std::same_as<bool> decltype(auto) result = (it == sent);
    assert(!result);
  }
  {
    View v(Iterator(array.data()), Sentinel(Iterator(array.data() + array.size())));
    std::ranges::filter_view view(std::move(v), [](auto) { return false; });
    auto const it = view.begin();
    auto const sent = view.end();
    std::same_as<bool> decltype(auto) result = (it == sent);
    assert(result);
  }
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>>();
  test<cpp20_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}

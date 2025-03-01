//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <utility>
#include <__type_traits/maybe_const.h>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"

#include "../types.h"

template <class Iterator>
constexpr void test() {
  using Sentinel   = sentinel_wrapper<Iterator>;
  using View       = minimal_view<Iterator, Sentinel>;
  using ConcatView = std::ranges::concat_view<View>;

  auto make_concat_view = [](auto begin, auto end) {
    View view{Iterator(begin), Sentinel(Iterator(end))};
    return ConcatView(std::move(view));
  };

  {
    std::array<int, 5> array{0, 1, 2, 3, 4};
    ConcatView view                          = make_concat_view(array.data(), array.data() + array.size());
    decltype(auto) it1                       = view.begin();
    decltype(auto) it2                       = view.begin();
    std::same_as<bool> decltype(auto) result = (it1 == it2);
    assert(result);

    ++it1;
    assert(!(it1 == it2));
  }

  {
    std::array<int, 5> array{0, 1, 2, 3, 4};
    ConcatView view = make_concat_view(array.data(), array.data() + array.size());
    assert(!(view.begin() == view.end()));
  }
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();
  test<cpp17_input_iterator<int const*>>();
  test<forward_iterator<int const*>>();
  test<bidirectional_iterator<int const*>>();
  test<random_access_iterator<int const*>>();
  test<contiguous_iterator<int const*>>();
  test<int const*>();

  return true;
}

int main(int, char**) {
  tests();
  return 0;
}

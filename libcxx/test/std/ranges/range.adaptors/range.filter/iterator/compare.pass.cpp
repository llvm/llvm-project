//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// friend constexpr bool operator==(iterator const&, iterator const&)
//  requires equality_comparable<iterator_t<V>>

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

template <class T>
concept has_equal = requires (T const& x, T const& y) { { x == y }; };

template <class Iterator>
constexpr void test() {
  using Sentinel = sentinel_wrapper<Iterator>;
  using View = minimal_view<Iterator, Sentinel>;
  using FilterView = std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = std::ranges::iterator_t<FilterView>;

  auto make_filter_view = [](auto begin, auto end, auto pred) {
    View view{Iterator(begin), Sentinel(Iterator(end))};
    return FilterView(std::move(view), pred);
  };

  {
    std::array<int, 5> array{0, 1, 2, 3, 4};
    FilterView view = make_filter_view(array.begin(), array.end(), AlwaysTrue{});
    FilterIterator it1 = view.begin();
    FilterIterator it2 = view.begin();
    std::same_as<bool> decltype(auto) result = (it1 == it2);
    assert(result);

    ++it1;
    assert(!(it1 == it2));
  }

  {
    std::array<int, 5> array{0, 1, 2, 3, 4};
    FilterView view = make_filter_view(array.begin(), array.end(), AlwaysTrue{});
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

  // Make sure `operator==` isn't provided for non comparable iterators
  {
    using Iterator = cpp20_input_iterator<int*>;
    using Sentinel = sentinel_wrapper<Iterator>;
    using FilterView = std::ranges::filter_view<minimal_view<Iterator, Sentinel>, AlwaysTrue>;
    using FilterIterator = std::ranges::iterator_t<FilterView>;
    static_assert(!has_equal<FilterIterator>);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}

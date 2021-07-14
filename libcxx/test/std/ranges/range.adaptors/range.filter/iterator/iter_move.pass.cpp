//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// friend constexpr range_rvalue_reference_t<V> iter_move(iterator const& i)
//  noexcept(noexcept(ranges::iter_move(i.current_)));

#include <ranges>

#include <array>
#include <cassert>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

template <class Iterator, bool HasNoexceptIterMove>
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
    FilterIterator const it = view.begin();

    int&& result = iter_move(it);
    static_assert(noexcept(iter_move(it)) == HasNoexceptIterMove);
    assert(&result == array.begin());
  }
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>,           /* noexcept */ false>();
  test<cpp20_input_iterator<int*>,           /* noexcept */ false>();
  test<forward_iterator<int*>,               /* noexcept */ false>();
  test<bidirectional_iterator<int*>,         /* noexcept */ false>();
  test<random_access_iterator<int*>,         /* noexcept */ false>();
  test<contiguous_iterator<int*>,            /* noexcept */ false>();
  test<int*,                                 /* noexcept */ true>();
  test<NoexceptIterMoveInputIterator<true>,  /* noexcept */ true>();
  test<NoexceptIterMoveInputIterator<false>, /* noexcept */ false>();
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}

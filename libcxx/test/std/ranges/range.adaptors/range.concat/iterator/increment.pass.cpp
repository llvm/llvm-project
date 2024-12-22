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
// constexpr iterator operator++(int) requires forward_range<V>;

#include <ranges>

#include <iostream>
#include <array>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"



template <class Iterator>
constexpr void test() {
  using Sentinel = sentinel_wrapper<Iterator>;
  using View = minimal_view<Iterator, Sentinel>;
  using ConcatView = std::ranges::concat_view<View>;
  using ConcatIterator = std::ranges::iterator_t<ConcatView>;

  auto make_concat_view = [](auto begin, auto end) {
    View view{Iterator(begin), Sentinel(Iterator(end))};
    return ConcatView(std::move(view));
  };

  // Increment an iterator when it won't find another satisfied value after begin()
  {
    std::array<int, 5> array{0, 1, 2, 3, 4};
    ConcatView view = make_concat_view(array.data(), array.data() + array.size());

    auto it = view.begin();
    auto& result = ++it;
    ASSERT_SAME_TYPE(ConcatIterator&, decltype(++it));
    assert(&result == &it);
    assert(*result == 1);
  }

  // Increment an iterator multiple times
  {
    std::array<int, 10> array{0,1,2,3,4};
    ConcatView view = make_concat_view(array.data(), array.data() + array.size());

    ConcatIterator it = view.begin();
    assert(*it == array[0]);

    ++it; assert(*it == array[1]);
    ++it; assert(*it == array[2]);
    ++it; assert(*it == array[3]);
    ++it; assert(*it == array[4]);
  }
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>   >();
  test<forward_iterator<int*>       >();
  test<bidirectional_iterator<int*> >();
  test<random_access_iterator<int*> >();
  test<contiguous_iterator<int*>    >();
  test<int*                         >();

  return true;
}

int main(int, char**) {
  tests();
  return 0;
}
